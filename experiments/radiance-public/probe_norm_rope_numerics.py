"""Model-free numeric checks for actual Qwen normalization and long-position RoPE.

Uses public checkpoint norm weights and random activations. No conversation,
token IDs or model-generated tensors are read or recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def run(root):
    import torch
    from safetensors import safe_open
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(4327)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model = Path(manifest.get('checkpoint', '/models/Qwen3.8-27B-Uncensored-MXFP4-awq'))
    config = json.loads((model / 'config.json').read_text())['text_config']
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    rows = []

    def error(actual, expected):
        actual, expected = actual.double(), expected.double()
        return float((actual - expected).norm() / expected.norm().clamp_min(1e-20))

    for name in ['model.language_model.layers.0.input_layernorm.weight',
                 'model.language_model.layers.3.self_attn.q_norm.weight',
                 'model.language_model.layers.3.self_attn.k_norm.weight']:
        with safe_open(model / index[name], framework='pt', device='cpu') as f:
            weight = f.get_tensor(name)
        width = weight.numel()
        norm = GemmaRMSNorm(width, eps=config['rms_norm_eps']).to(device='cuda', dtype=torch.bfloat16)
        norm.weight.copy_(weight.cuda())
        for tokens in [1, 7, 8, 9, 63, 64, 65, 2047, 2048, 2049]:
            for amplitude in [.001, 1, 10000]:
                for add_residual in [False, True]:
                    x = (torch.randn((tokens, width), device='cuda') * amplitude).bfloat16()
                    residual = (torch.randn_like(x.float()) * amplitude).bfloat16() if add_residual else None
                    reference_input = x.double() + (residual.double() if residual is not None else 0)
                    expected = reference_input * torch.rsqrt(reference_input.square().mean(-1, keepdim=True) + config['rms_norm_eps'])
                    expected *= norm.weight.double() + 1
                    actual = norm(x.clone(), residual.clone() if residual is not None else None)
                    output = actual[0] if add_residual else actual
                    row = dict(kind='norm',name=name,tokens=tokens,amplitude=amplitude,
                               residual=add_residual,relative_error=error(output,expected),
                               finite=bool(torch.isfinite(output).all()))
                    if add_residual:
                        row['residual_relative_error'] = error(actual[1],reference_input)
                    rows.append(row)

    parameters = config.get('rope_parameters', config.get('rope_scaling'))
    if parameters is None:
        raise ValueError('reference config has no explicit RoPE parameters')
    head = config['head_dim']; rotary = int(head * parameters['partial_rotary_factor'])
    with torch.device('cuda'):
        rope = MRotaryEmbedding(head,rotary,config['max_position_embeddings'],parameters['rope_theta'],
                               True,torch.bfloat16,mrope_section=parameters['mrope_section'],
                               mrope_interleaved=parameters['mrope_interleaved'])
    def expected_rope(value, positions):
        inv = torch.pow(torch.tensor(float(parameters['rope_theta']), device='cuda', dtype=torch.float64),
                        -torch.arange(0,rotary,2,device='cuda',dtype=torch.float64)/rotary)
        angle = positions.double().view(-1,1) * inv
        cos, sin = angle.cos()[:,None,:], angle.sin()[:,None,:]
        value = value.view(positions.numel(),-1,head).double()
        first,second = value[...,:rotary//2],value[...,rotary//2:rotary]
        result=torch.cat([first*cos-second*sin,second*cos+first*sin,value[...,rotary:]],dim=-1)
        return result.reshape(positions.numel(),-1)
    for start in [0,32760,60000,86142,132739,200000,253720]:
        for tokens in [1,7,8,9,63,64,65]:
            positions=torch.arange(start,start+tokens,device='cuda',dtype=torch.long)
            query=torch.randn((tokens,config['num_attention_heads']*head),device='cuda',dtype=torch.bfloat16)
            key=torch.randn((tokens,config['num_key_value_heads']*head),device='cuda',dtype=torch.bfloat16)
            expected_q,expected_k=expected_rope(query,positions),expected_rope(key,positions)
            for rows_3 in [False,True]:
                actual_q,actual_k=rope(positions.repeat(3,1) if rows_3 else positions,query.clone(),key.clone())
                rows.append(dict(kind='rope',position=start,tokens=tokens,three_position_rows=rows_3,
                                 relative_error=max(error(actual_q,expected_q),error(actual_k,expected_k)),
                                 finite=bool(torch.isfinite(actual_q).all() and torch.isfinite(actual_k).all()),
                                 unrotated_channels_exact=bool(torch.equal(actual_q.view(tokens,-1,head)[...,rotary:],query.view(tokens,-1,head)[...,rotary:]))))
    report={'cases':rows,'all_finite':all(r['finite'] for r in rows),
            'maximum_relative_error':max(r['relative_error'] for r in rows),
            'actual_norm_dispatch':norm._forward_method.__name__,
            'actual_rope_dispatch':rope._forward_method.__name__,
            'rope_parameters':parameters,
            'module_hashes':{str(Path(__import__(cls.__module__,fromlist=['__file__']).__file__).name):hashlib.sha256(Path(__import__(cls.__module__,fromlist=['__file__']).__file__).read_bytes()).hexdigest() for cls in [GemmaRMSNorm,MRotaryEmbedding]}}
    (root/'norm-rope-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    if (not report['all_finite'] or report['maximum_relative_error']>.02
            or any(not r.get('unrotated_channels_exact',True) for r in rows)):
        raise ValueError('normalization/position encoding deviates from independent reference')


if __name__=='__main__':
    from vllm.config import VllmConfig, set_current_vllm_config

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    config = VllmConfig()
    config.compilation_config.custom_ops = ['all']
    with set_current_vllm_config(config):
        run(parser.parse_args().root)
