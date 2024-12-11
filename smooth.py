import torch
from torch import nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
          attn_ln = module.self_attn_layer_norm
          qkv = [
              module.self_attn.q_proj,
              module.self_attn.k_proj,
              module.self_attn.v_proj,
          ]
          qkv_input_scales = scales[name + ".self_attn.q_proj"]
          smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

          ffn_ln = module.final_layer_norm
          fc1 = module.fc1
          fc1_input_scales = scales[name + ".fc1"]
          smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

# smoothing with different alphas for attn vs fc 
# alpha=-1 to not smooth
@torch.no_grad()
def smooth_lm_two_alphas(model, scales, alpha_qkv=0.5, alpha_fc=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
          attn_ln = module.self_attn_layer_norm
          qkv = [
              module.self_attn.q_proj,
              module.self_attn.k_proj,
              module.self_attn.v_proj,
          ]
          qkv_input_scales = scales[name + ".self_attn.q_proj"]
          if alpha_qkv > 0:
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha_qkv)

          ffn_ln = module.final_layer_norm
          fc1 = module.fc1
          fc1_input_scales = scales[name + ".fc1"]
          if alpha_fc > 0:
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha_fc)