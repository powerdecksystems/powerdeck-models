"""export.py — HF model → ONNX artifact + parity check (BP196 §9).

Runs in the export workflow (or any build box). Recipes:

  seq2seq  — optimum ORTModelForSeq2SeqLM (text encoder-decoders)
  chronos  — chronos-forecasting pipeline, torch.onnx.export of the
             inner model on a representative context tensor

    python tools/export.py --model amazon/chronos-2 --name chronos-2 \
        --recipe chronos --out dist/chronos-2

Writes <out>/model.onnx and <out>/fragment.json (manifest entry minus
`url`, which manifest_update.py fills in at release time).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _parity(torch_out, onnx_path: Path, inputs: dict, atol: float) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                providers=['CPUExecutionProvider'])
    got = sess.run(None, inputs)
    diff = float(np.max(np.abs(np.asarray(torch_out) - got[0])))
    if diff > atol:
        raise SystemExit(
            f"PARITY FAILED: max |Δ| = {diff} > {atol} — not publishing")
    return diff


def _export_seq2seq(model: str, out: Path, atol: float) -> tuple[float, dict]:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    m = ORTModelForSeq2SeqLM.from_pretrained(model, export=True)
    m.save_pretrained(out)
    tm = AutoModelForSeq2SeqLM.from_pretrained(model)
    tok = AutoTokenizer.from_pretrained(model)
    enc = tok('parity check input', return_tensors='pt')
    with torch.no_grad():
        tout = tm(**enc).logits.numpy()
    inputs = {k: v.numpy() for k, v in enc.items()}
    return _parity(tout, out / 'model.onnx', inputs, atol), inputs


def _export_chronos(model: str, out: Path, atol: float,
                    horizon_patches: int = 8) -> tuple[float, dict]:
    """chronos-forecasting models — torch.onnx.export the inner
    encoder-decoder on a representative context window."""
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model, device_map='cpu')
    inner = pipe.model.eval()

    class _Wrap(torch.nn.Module):
        """Chronos2Output → quantile_preds (batch, quantiles, horizon)."""
        def __init__(self, m, horizon_patches):
            super().__init__()
            self.m = m
            self.h = horizon_patches

        def forward(self, context):
            return self.m(context,
                          num_output_patches=self.h).quantile_preds

    w = _Wrap(inner, horizon_patches).eval()
    ctx = torch.sin(torch.linspace(0, 6.28, 128))[None, :]  # 1×128 context
    with torch.no_grad():
        tout = w(ctx).numpy()
    onnx_path = out / 'model.onnx'
    torch.onnx.export(
        w, (ctx,), onnx_path,
        input_names=['context'], output_names=['forecast'],
        dynamic_axes={'context': {0: 'batch', 1: 'length'},
                      'forecast': {0: 'batch'}},
        opset_version=17)
    return _parity(tout, onnx_path, {'context': ctx.numpy()}, atol), \
        {'context_shape': list(ctx.shape)}


RECIPE = {'seq2seq': _export_seq2seq, 'chronos': _export_chronos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='HF repo or local path')
    ap.add_argument('--name', required=True, help='manifest artifact name')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--recipe', default='seq2seq',
                    choices=sorted(RECIPE), help='export recipe')
    ap.add_argument('--atol', type=float, default=1e-4)
    ap.add_argument('--title', default='', help='display name (picker)')
    ap.add_argument('--description', default='',
                    help='what the model is (picker)')
    ap.add_argument('--use-case', default='',
                    help='best use case, shown to the user (picker)')
    ap.add_argument('--tier', default='ai',
                    help='license tier label shown in the picker')
    ap.add_argument('--kind', default='',
                    help='step family — forecast/anomaly/battery/…')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    diff, meta = RECIPE[args.recipe](args.model, args.out, args.atol)

    onnx_path = args.out / 'model.onnx'
    frag = {
        'name': args.name,
        'title': args.title or args.name,
        'description': args.description,
        'use_case': args.use_case,
        'tier': args.tier,
        'kind': args.kind,
        'format': 'onnx',
        'artifact': f'{args.name}/model.onnx',
        'sha256': _sha256(onnx_path),
        'bytes': onnx_path.stat().st_size,
        'source_model': args.model,
        'parity_max_abs_diff': diff,
        **(meta if isinstance(meta, dict) else {}),
    }
    (args.out / 'fragment.json').write_text(json.dumps(frag, indent=2))
    print(f"exported {args.model} → {onnx_path} "
          f"(parity Δ={diff:.2e}, sha256={frag['sha256'][:12]}…)")
    print('manifest fragment:', args.out / 'fragment.json')


if __name__ == '__main__':
    main()
