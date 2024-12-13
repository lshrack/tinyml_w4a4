import torch
from torch import nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from quantize import pseudo_quantize_tensor

@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, name, w_bit,
                     q_group_size,
                     input_feat, kwargs = {}):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)
        best_error = float('inf')
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid
            scales = (s_x ** ratio).clamp(min=1e-4)
            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:

                scales = scales.to(fc.weight.device)

                # Scale up the values of the weight channels
                fc.weight.mul_(scales)
                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)
                fc.weight.div_(scales)

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    if isinstance(module, OPTDecoderLayer):
      # attention input
      inp = input_feat[name + '.self_attn.q_proj'] # out_proj => q_proj
      inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
      qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
      final_scales = _search_module_scale(module.self_attn, qkv, inp)
      scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

      # attn out
      inp = input_feat[name + '.self_attn.out_proj']
      inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
      final_scales = _search_module_scale(module.self_attn.out_proj, [module.self_attn.out_proj], inp)
      scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

      # fc1
      inp = input_feat[name + '.fc1']
      inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
      final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
      scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

      # fc2
      inp = input_feat[name + '.fc2']
      inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
      final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
      scale_fc_fc(module.fc1, module.fc2, final_scales)

    elif isinstance(module, LlamaDecoderLayer):
      print("scaling llama", name)
      # attention input
      inp = input_feat[name + '.self_attn.q_proj']
      inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
      qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
      final_scales = _search_module_scale(module.self_attn, qkv, inp, kwargs)
      scale_ln_fcs(module.input_layernorm, qkv, final_scales)

      # attn out
      if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        inp = input_feat[name + '.self_attn.o_proj']
        inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
        final_scales = _search_module_scale(module.self_attn.o_proj, [module.self_attn.o_proj], inp)
        scale_fc_fc(module.self_attn.v_proj, module.self_attn.o_proj, final_scales)
      else:
        print("skipping attn out")

      # fc1
      inp = input_feat[name + '.mlp.gate_proj']
      inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
      layers = [module.mlp.gate_proj, module.mlp.up_proj]
      final_scales = _search_module_scale(module.mlp, layers, inp)
      scale_ln_fcs(module.post_attention_layernorm, layers, final_scales)

      # fc2
      inp = input_feat[name + ".mlp.down_proj"]
      inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
      final_scales = _search_module_scale(module.mlp.down_proj, [module.mlp.down_proj], inp)
      scale_fc_fc(module.mlp.up_proj, module.mlp.down_proj, final_scales)
    

def awq_scale_model(model, input_feat, w_bit, q_group_size):
  for name, module in model.named_modules():
        if isinstance(module, (OPTDecoderLayer, LlamaDecoderLayer)):
          auto_scale_block(module, name, w_bit, q_group_size, input_feat, kwargs={"position_ids": torch.arange(0, 124, dtype=torch.long, device="cuda").unsqueeze(0)})
