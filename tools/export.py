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
                    horizon_patches: int = 8,
                    context_len: int = 512) -> tuple[float, dict]:
    """chronos-forecasting models — torch.onnx.export the inner
    encoder-decoder on a representative context window."""
    import torch
    from chronos import BaseChronosPipeline
    from torch.onnx import register_custom_op_symbolic

    def _nanmean(g, x, dim, keepdim=False, dtype=None):
        """nanmean(x,dim) = sum(nan_to_num(x),dim) / sum(~isnan(x),dim) —
        aten::nanmean has no ONNX symbolic."""
        from torch.onnx import symbolic_helper as sh
        axes = sh._parse_arg(dim, 'is')  # traced Constant → list[int]
        isnan = g.op("IsNaN", x)
        inv = g.op("Not", isnan)
        zero = g.op("Constant",
                    value_t=torch.zeros(1, dtype=torch.float32))
        clean = g.op("Where", isnan, zero, x)
        kd = 1 if sh._parse_arg(keepdim, 'i') else 0
        num = g.op("ReduceSum", clean, axes_i=axes, keepdims_i=kd)
        cnt = g.op("ReduceSum",
                   g.op("Cast", inv, to_i=1),  # bool → float32
                   axes_i=axes, keepdims_i=kd)
        return g.op("Div", num, cnt)

    def _asinh(g, x):
        """asinh(x) = ln(x + sqrt(x² + 1)) — no opset-17 symbolic either
        (chronos_bolt InstanceNorm inverse transform)."""
        one = g.op("Constant",
                   value_t=torch.ones(1, dtype=torch.float32))
        return g.op("Log", g.op(
            "Add", x, g.op("Sqrt", g.op("Add", g.op("Mul", x, x), one))))

    def _sinh(g, x):
        """sinh(x) = (eˣ − e⁻ˣ)/2 — used by Chronos2RotaryEmbedding."""
        two = g.op("Constant",
                   value_t=torch.full((1,), 2.0, dtype=torch.float32))
        return g.op("Div", g.op("Sub", g.op("Exp", x),
                                g.op("Exp", g.op("Neg", x))), two)

    def _cosh(g, x):
        """cosh(x) = (eˣ + e⁻ˣ)/2 — paired with sinh in rope embed."""
        two = g.op("Constant",
                   value_t=torch.full((1,), 2.0, dtype=torch.float32))
        return g.op("Div", g.op("Add", g.op("Exp", x),
                                g.op("Exp", g.op("Neg", x))), two)

    def _atanh(g, x):
        """atanh(x) = ½ ln((1+x)/(1−x)) — head off the whole
        hyperbolic family in one pass rather than per-run."""
        half = g.op("Constant",
                    value_t=torch.full((1,), 0.5, dtype=torch.float32))
        one = g.op("Constant",
                   value_t=torch.ones(1, dtype=torch.float32))
        return g.op("Mul", half, g.op("Log", g.op(
            "Div", g.op("Add", one, x), g.op("Sub", one, x))))

    for op, fn in (("aten::nanmean", _nanmean), ("aten::asinh", _asinh),
                   ("aten::sinh", _sinh), ("aten::cosh", _cosh),
                   ("aten::atanh", _atanh)):
        register_custom_op_symbolic(op, fn, 17)

    pipe = BaseChronosPipeline.from_pretrained(model, device_map='cpu')
    inner = pipe.model.eval()

    class _EmbedCast(torch.nn.Module):
        """torch.full without dtype= exports as ConstantOfShape whose
        value attr is float-typed — the [REG] token ids then reach
        Gather as float, which ORT rejects. Cast ids at the boundary."""
        def __init__(self, emb):
            super().__init__()
            self.emb = emb

        def forward(self, ids):
            return self.emb(ids.long())

    inner.shared = _EmbedCast(inner.shared)

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
    ctx = torch.sin(torch.linspace(0, 6.28, context_len))[None, :]
    with torch.no_grad():
        tout = w(ctx).numpy()
    onnx_path = out / 'model.onnx'
    # Static shapes only — chronos's patch embed uses Tensor.unfold,
    # which cannot export against a symbolic input length. Consumers
    # pad/truncate to fragment context_length.
    torch.onnx.export(
        w, (ctx,), onnx_path,
        input_names=['context'], output_names=['forecast'],
        opset_version=17, dynamo=False)  # dynamo exporter emits an
    # invalid bool-input Einsum for this model — legacy tracer is clean
    return _parity(tout, onnx_path, {'context': ctx.numpy()}, atol), \
        {'context_length': context_len,
         'horizon_steps': horizon_patches * getattr(
             inner.chronos_config, 'output_patch_size', 16)}


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
