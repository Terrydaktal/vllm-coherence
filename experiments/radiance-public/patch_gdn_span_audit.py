"""Diagnostic-only GDN decay-range counters; no tensor values are recorded.

The probe synchronizes the GPU while examining prefill gates. It is enabled only
for named isolated requests and must never be installed in the production lane.
"""

import hashlib
from pathlib import Path

PREIMAGE = "f6675ecfcc0bba7c8f2b77db91438e6f6888a251d7140b443d51f3c21aa2dbe6"
ANCHOR = "    return q, k, v, g, beta\n"
HELPER = '''

# Offline investigation only: aggregate counts, never dump gate tensors.
_qwen_decay_audit = {}


def _qwen_record_decay_range(g, num_seqs, tokens):
    import json as _json
    import pathlib as _pathlib
    import time as _time
    marker = _pathlib.Path('/benchmark/gdn-audit-request.json')
    if torch.cuda.is_current_stream_capturing() or not marker.exists():
        return
    request = _json.loads(marker.read_text())['label']
    row = _qwen_decay_audit.setdefault(request, {
        'invocations': 0, 'skipped_multi_sequence_invocations': 0,
        'chunk_heads': 0, 'chunk_heads_exceeding_span_160': 0,
        'maximum_decay_span': 0.0, 'observed_token_rows_across_layers': 0,
    })
    row['invocations'] += 1
    if num_seqs != 1:
        row['skipped_multi_sequence_invocations'] += 1
    else:
        heads = g.shape[-1]
        full, tail = divmod(tokens, CHUNK)
        spans = []
        if full:
            chunks = g[:full * CHUNK].view(full, CHUNK, heads)
            spans.append((chunks[:, -1] - chunks[:, 0]).abs().flatten())
        if tail:
            spans.append((g[-1] - g[-tail]).abs().flatten())
        values = torch.cat(spans)
        row['chunk_heads'] += values.numel()
        row['chunk_heads_exceeding_span_160'] += int((values > 160).sum().item())
        row['maximum_decay_span'] = max(row['maximum_decay_span'], float(values.max().item()))
        row['observed_token_rows_across_layers'] += tokens
    report = {'at': _time.time(), 'intrusive_gpu_synchronization': True,
              'requests': _qwen_decay_audit}
    output = _pathlib.Path('/benchmark/gdn-decay-audit.json')
    temporary = output.with_suffix('.tmp')
    temporary.write_text(_json.dumps(report))
    temporary.replace(output)
'''


def install(package):
    path = Path(package) / "radiance_gdn.py"
    source = path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PREIMAGE
    assert source.count(ANCHOR) == 1
    updated = source.replace(ANCHOR, "    _qwen_record_decay_range(g, num_seqs, T)\n" + ANCHOR) + HELPER
    compile(updated, str(path), "exec")
    path.write_text(updated)
    return {"source_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "diagnostic_only": True, "records_tensor_values": False}
