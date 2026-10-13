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
        # opset ≥18: ReduceSum takes axes as an INPUT, not attribute
        axes_c = g.op("Constant",
                      value_t=torch.tensor(axes, dtype=torch.int64))
        num = g.op("ReduceSum", clean, axes_c, keepdims_i=kd)
        cnt = g.op("ReduceSum",
                   g.op("Cast", inv, to_i=1),  # bool → float32
                   axes_c, keepdims_i=kd)
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

    pipe = (model if isinstance(model, BaseChronosPipeline)
            else BaseChronosPipeline.from_pretrained(model,
                                                     device_map='cpu'))
    inner = pipe.model.eval().to('cpu')  # fit() may return it on cuda

    class _EmbedCast(torch.nn.Module):
        """torch.full without dtype= exports as ConstantOfShape whose
        value attr is float-typed — the [REG] token ids then reach
        Gather as float, which ORT rejects. Cast ids at the boundary."""
        def __init__(self, emb):
            super().__init__()
            self.emb = emb

        def forward(self, ids):
            return self.emb(ids.long())

    # wrap shared/embed_tokens at EVERY nesting level — the merged
    # (post-LoRA) model looks ids up via model.shared, not just the
    # top-level alias that worked for the base model
    for _pname, parent in inner.named_modules():
        for _cname, child in parent.named_children():
            if _cname in ('shared', 'embed_tokens') and \
                    isinstance(child, torch.nn.Embedding):
                setattr(parent, _cname, _EmbedCast(child))

    # The legacy tracer mis-shapes two constructs inside
    # _prepare_patched_context: Tensor.unfold (patching) and the einops
    # repeat (time encoding) — both emit transposed dims that fail ORT
    # shape inference (and would silently corrupt the model if they
    # didn't). Reimplement with unfold→reshape (patch_stride ==
    # patch_size here, so unfold is exactly a reshape) and
    # einops→expand. Verified against the unpatched forward below.
    import types as _types
    cfg = inner.chronos_config

    def _prep_patched_context(self, context, context_mask=None):
        if context_mask is not None:
            context_mask = context_mask.to(context.dtype)
        else:
            context_mask = torch.isnan(context).logical_not().to(
                context.dtype)
        b, length = context.shape
        if length > cfg.context_length:
            context = context[..., -cfg.context_length:]
            context_mask = context_mask[..., -cfg.context_length:]
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)
        p, stride = cfg.input_patch_size, cfg.input_patch_stride
        pad = (-context.shape[-1]) % stride
        if pad:
            nan_pad = torch.full((*context.shape[:-1], pad),
                              torch.nan, dtype=context.dtype,
                              device=context.device)
            context = torch.cat([nan_pad, context], dim=-1)
            m_pad = torch.zeros((*context_mask.shape[:-1], pad),
                             dtype=context_mask.dtype,
                             device=context_mask.device)
            context_mask = torch.cat([m_pad, context_mask], dim=-1)
        n = (context.shape[-1] - p) // stride + 1
        if stride == p:
            patched = context.reshape(b, n, p)
            patched_mask = context_mask.reshape(b, n, p)
        else:  # strided overlap — stride≠p can't be a plain reshape
            idx = (torch.arange(p, device=context.device)[None]
                   + stride * torch.arange(n, device=context.device)[:, None])
            patched = context[:, idx]
            patched_mask = context_mask[:, idx]
        patched_mask = torch.nan_to_num(patched_mask, nan=0.0)
        patched = torch.where(patched_mask > 0.0, patched, 0.0)
        attention_mask = patched_mask.sum(dim=-1) > 0
        tenc = torch.arange(-(n * p), 0, device=context.device,
                         dtype=torch.float32)
        tenc = (tenc.reshape(1, n, p).expand(b, n, p)
                .div(int(cfg.time_encoding_scale)).to(self.dtype))
        return (torch.cat([tenc, patched, patched_mask], dim=-1),
                attention_mask, loc_scale)

    ctx = torch.sin(torch.linspace(0, 6.28, context_len))[None, :]
    with torch.no_grad():
        ref = inner(ctx, num_output_patches=horizon_patches).quantile_preds
    # apply to every module that OWNS the method — the merged
    # (post-LoRA) model's forward delegates to a nested module whose
    # own _prepare_patched_context would otherwise run unpatched
    patched_ctx = 0
    for mod in inner.modules():
        if hasattr(mod, '_prepare_patched_context'):
            mod._prepare_patched_context = _types.MethodType(
                _prep_patched_context, mod)
            patched_ctx += 1
    print(f'patched _prepare_patched_context on {patched_ctx} modules')

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
    with torch.no_grad():
        tout = w(ctx).numpy()
    delta = float(abs(ref.numpy() - tout).max())
    assert delta < 1e-4, \
        f'patched _prepare_patched_context diverges from original: {delta}'
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

    write_fragment(args.out, name=args.name, source=args.model,
                   diff=diff, meta=meta, title=args.title,
                   description=args.description, use_case=args.use_case,
                   tier=args.tier, kind=args.kind)
    print(f"exported {args.model} → {args.out / 'model.onnx'}")


def write_fragment(out: Path, *, name: str, source: str, diff: float,
                   meta: dict, title: str = '', description: str = '',
                   use_case: str = '', tier: str = 'ai',
                   kind: str = '') -> Path:
    """Emit <out>/fragment.json — the manifest entry minus `url`, which
    manifest_update.py fills at release time. Shared by export.py and
    train.py so both pipelines produce identical catalog entries."""
    onnx_path = out / 'model.onnx'
    frag = {
        'name': name,
        'title': title or name,
        'description': description,
        'use_case': use_case,
        'tier': tier,
        'kind': kind,
        'format': 'onnx',
        'artifact': f'{name}/model.onnx',
        'sha256': _sha256(onnx_path),
        'bytes': onnx_path.stat().st_size,
        'source_model': source,
        'parity_max_abs_diff': diff,
        **(meta if isinstance(meta, dict) else {}),
    }
    frag_path = out / 'fragment.json'
    frag_path.write_text(json.dumps(frag, indent=2))
    print(f"fragment: {frag_path} "
          f"(parity Δ={diff:.2e}, sha256={frag['sha256'][:12]}…)")
    return frag_path


if __name__ == '__main__':
    main()
