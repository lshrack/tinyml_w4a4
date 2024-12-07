from functools import partial
import torch
from torch import nn
from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
)

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=4):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=4):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=4):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=4):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_group(t, n_bits=4):
    return pseudo_quantize_tensor(t, n_bits, q_group_size=128)


class WALinear(nn.Module):
    def __init__(
        self,
        n_bits,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=n_bits
            )
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=n_bits
            )
        elif act_quant == "per_group":
            self.act_quant_name = "per_group"
            self.act_quant = partial(quantize_activation_per_group, n_bits=n_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WALinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = y
        # q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        n_bits,
        module,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False,
        group_size=-1,
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = WALinear(
            n_bits,
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=n_bits
            )  # use 4-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=n_bits
            )
        elif weight_quant == "per_group" and group_size > 0:
            new_module.weight = pseudo_quantize_tensor(
                module.weight, n_bit=n_bits, q_group_size=group_size
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"WALinear({self.n_bits}, {self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model,
    n_bits,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=True,
    group_size=-1,
):
    for _, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = WALinear.from_float(
                n_bits,
                m.fc1,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
            )
            m.fc2 = WALinear.from_float(
                n_bits,
                m.fc2,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
            )
        elif isinstance(m, OPTAttention):
            # Here we simulate quantizing BMM inputs by quantizing the
            # output of q_proj, k_proj, v_proj
            m.q_proj = WALinear.from_float(
                n_bits,
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                group_size=group_size,
            )
            m.k_proj = WALinear.from_float(
                n_bits,
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                group_size=group_size,
            )
            m.v_proj = WALinear.from_float(
                n_bits,
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                group_size=group_size,
            )
            m.out_proj = WALinear.from_float(
                n_bits,
                m.out_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
            )
    model.lm_head = WALinear.from_float(
        n_bits,
        model.lm_head,
        weight_quant=weight_quant,
        act_quant=act_quant,
        quantize_output=quantize_bmm_input,
        group_size=group_size,
    )
    return model


def quantize_llama(
    model,
    n_bits,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=True,
    group_size=-1,
):
    for _, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            m.gate_proj = WALinear.from_float(
                n_bits,
                m.gate_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
            )
            m.up_proj = WALinear.from_float(
                n_bits,
                m.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
            )
            m.down_proj = WALinear.from_float(
                n_bits,
                m.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
            )
        elif isinstance(m, LlamaAttention):
            # Here we simulate quantizing BMM inputs by quantizing the
            # output of q_proj, k_proj, v_proj
            m.q_proj = WALinear.from_float(
                n_bits,
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                group_size=group_size,
            )
            m.k_proj = WALinear.from_float(
                n_bits,
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                group_size=group_size,
            )
            m.v_proj = WALinear.from_float(
                n_bits,
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                group_size=group_size,
            )
            m.o_proj = WALinear.from_float(
                n_bits,
                m.o_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
            )
    model.lm_head = WALinear.from_float(
        n_bits,
        model.lm_head,
        weight_quant=weight_quant,
        act_quant=act_quant,
        quantize_output=quantize_bmm_input,
        group_size=group_size,
    )
    return model
