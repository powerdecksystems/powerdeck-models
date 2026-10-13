"""train.py — generic model training → ONNX artifact (BP198 §2/§4).

Trains on public or synthetic data ONLY — customer telemetry never
enters this pipeline. Each recipe produces <out>/model.onnx plus the
same fragment.json contract as export.py (write_fragment is shared).

  pq        — IEEE-1159 synthetic waveforms → 1D-CNN classifier
  anomaly   — kkonathala/powerplant-anomaly-transformer trained
              self-supervised on synthetic normal telemetry
  ttm       — EnergyFM/energy-ttm fine-tune on UCI appliances energy
  chronos   — Chronos2Pipeline.fit() LoRA on public energy data
              (GPU recipe — Kaggle; runs on CPU but slow)

    python tools/train.py --recipe pq --name pq-classifier \
        --out dist/pq-classifier --epochs 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pd_datasets import (PQ_CLASSES, ieee1159_waveforms,
                       synthetic_telemetry, uci_appliances_energy,
                       nasa_liion_capacities)
from export import _parity, _sha256, write_fragment


# ── pq-classifier ────────────────────────────────────────────────────

def _train_pq(args, out: Path) -> tuple[float, dict]:
    """8-class IEEE-1159 waveform classifier — small 1D-CNN trained on
    fully synthetic data; the only catalog model with no HF source."""
    import torch
    from torch import nn

    X, y = ieee1159_waveforms(n_per_class=args.samples_per_class,
                             window=args.window, seed=args.seed)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    net = nn.Sequential(
        nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.ReLU(),
        nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
        nn.AdaptiveAvgPool1d(16), nn.Flatten(),
        nn.Linear(32 * 16, 128), nn.ReLU(),
        nn.Linear(128, len(PQ_CLASSES)),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(X_t)
    net.train()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = correct = 0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            logits = net(X_t[idx])
            loss = lossf(logits, y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += (logits.argmax(1) == y_t[idx]).sum().item()
        acc = tot / n
        print(f'epoch {ep + 1}/{args.epochs} acc={acc:.3f}')
    net.eval()

    sample = X_t[:1]
    with torch.no_grad():
        tout = net(sample).numpy()
    onnx_path = out / 'model.onnx'
    torch.onnx.export(net, (sample,), onnx_path,
                      input_names=['waveform'],
                      output_names=['logits'],
                      dynamic_axes={'waveform': {0: 'batch'},
                                    'logits': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    meta = {'classes': PQ_CLASSES, 'window': args.window,
            'source_dataset': 'synthetic-ieee1159',
            'train_acc': round(acc, 4)}
    return _parity(tout, onnx_path,
                   {'waveform': sample.numpy()}, args.atol), meta


# ── anomaly-transformer ──────────────────────────────────────────────

def _train_anomaly(args, out: Path) -> tuple[float, dict]:
    """kkonathala/powerplant-anomaly-transformer — self-supervised
    training on synthetic normal telemetry (the repo ships code only;
    we produce the weights)."""
    import subprocess
    import torch
    from torch.nn import functional as F

    src = out / '_src'
    if not src.exists():
        subprocess.check_call(
            ['git', 'clone', '--depth', '1',
             'https://huggingface.co/kkonathala/'
             'powerplant-anomaly-transformer', str(src)])
    sha = subprocess.check_output(
        ['git', '-C', str(src), 'rev-parse', 'HEAD'],
        text=True).strip()
    sys.path.insert(0, str(src))
    from model import PowerPlantAnomalyTransformer  # noqa: E402

    cfg = json.loads((src / 'config.json').read_text())['architecture']
    cfg['n_channels'] = args.channels
    m = PowerPlantAnomalyTransformer(**{k: v for k, v in cfg.items()
                                        if k != 'num_patches'
                                        and k != 'total_parameters'})
    data = synthetic_telemetry(n_windows=args.windows,
                               channels=args.channels,
                               length=cfg['context_length'],
                               seed=args.seed)
    X = torch.from_numpy(data)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    m.train()
    for ep in range(args.epochs):
        perm = torch.randperm(len(X))
        ep_loss = 0.0
        for i in range(0, len(X), 16):
            res = m(X[perm[i:i + 16]], training=True)
            loss = res['loss']
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        print(f'epoch {ep + 1}/{args.epochs} '
              f'loss={ep_loss / (len(X) // 16):.4f}')
    m.eval()

    class _Infer(torch.nn.Module):
        """inference path only — anomaly_scores + reconstruction."""
        def __init__(self, model):
            super().__init__()
            self.m = model

        def forward(self, x):
            r = self.m(x, training=False)
            return r['anomaly_scores'], r['reconstruction']

    w = _Infer(m).eval()
    sample = X[:1]
    with torch.no_grad():
        tout = w(sample)[0].numpy()
    onnx_path = out / 'model.onnx'
    torch.onnx.export(
        w, (sample,), onnx_path,
        input_names=['series'],
        output_names=['anomaly_scores', 'reconstruction'],
        dynamic_axes={'series': {0: 'batch'},
                      'anomaly_scores': {0: 'batch'},
                      'reconstruction': {0: 'batch'}},
        opset_version=17, dynamo=False)
    meta = {'channels': args.channels,
            'context_length': cfg['context_length'],
            'source_dataset': 'synthetic-telemetry',
            'source_commit': sha}
    return _parity(tout, onnx_path, {'series': sample.numpy()},
                   args.atol), meta


# ── energy-ttm ───────────────────────────────────────────────────────

def _train_ttm(args, out: Path) -> tuple[float, dict]:
    """EnergyFM/energy-ttm — optional fine-tune on UCI appliances data,
    then ONNX export on static context."""
    import pandas as pd
    import torch
    try:
        from transformers import TinyTimeMixerForPrediction
    except ImportError:  # TTM moved to IBM's tsfm package
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerForPrediction)

    m = TinyTimeMixerForPrediction.from_pretrained(args.model)
    ctx_len = m.config.context_length
    horizon = m.config.prediction_length

    if args.epochs > 0:
        df = pd.read_csv(uci_appliances_energy())
        series = torch.tensor(
            df['Appliances'].values, dtype=torch.float32)
        # (1, T, 1) univariate — sliding windows are handled internally
        seq = series[None, :, None]
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        total = ctx_len + horizon
        for ep in range(args.epochs):
            ep_loss = nb = 0
            for i in range(0, seq.shape[1] - total, args.batch_size):
                idx = torch.arange(i, i + args.batch_size)
                idx = idx[idx + total <= seq.shape[1]]
                if not len(idx):
                    break
                past = torch.stack(
                    [seq[0, j:j + ctx_len, 0] for j in idx])[..., None]
                fut = torch.stack(
                    [seq[0, j + ctx_len:j + total, 0] for j in idx]
                    )[..., None]
                r = m(past_values=past, future_values=fut)
                loss = r.loss if getattr(r, 'loss', None) is not None \
                    else torch.nn.functional.mse_loss(
                        r.prediction_outputs, fut)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
            print(f'epoch {ep + 1}/{args.epochs} '
                  f'loss={ep_loss / max(nb, 1):.4f}')
    m.eval()

    # TinyTimeMixerPatchify uses Tensor.unfold — the legacy tracer
    # transposes its output dims (same bug class as chronos's patch
    # embed). Replace every patchify instance with a gather-based
    # equivalent; verified against the unpatched forward below.
    import types as _types

    def _patchify_gather(self, past_values):
        if past_values.shape[-2] != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({past_values.shape[-2]}) "
                f"doesn't match model configuration "
                f"({self.sequence_length}).")
        x = past_values[:, self.sequence_start:, :]
        b = x.shape[0]
        p, s, n = self.patch_length, self.patch_stride, self.num_patches
        idx = (torch.arange(p, device=x.device)[None, :]
               + s * torch.arange(n, device=x.device)[:, None])
        out = x[:, idx.reshape(-1), :]
        out = out.reshape(b, n, p, x.shape[-1])
        return out.permute(0, 3, 1, 2).contiguous()

    class _Wrap(torch.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.m = mm

        def forward(self, past):
            return self.m(past_values=past).prediction_outputs

    w = _Wrap(m).eval()
    sample = torch.sin(torch.linspace(0, 6.28 * 10, ctx_len)
                       )[None, :, None]
    with torch.no_grad():
        ref = w(sample).numpy()
    patched = 0
    for mod in m.modules():
        if type(mod).__name__ == 'TinyTimeMixerPatchify':
            mod.forward = _types.MethodType(_patchify_gather, mod)
            patched += 1
    with torch.no_grad():
        tout = w(sample).numpy()
    delta = float(abs(ref - tout).max())
    assert delta < 1e-4, \
        f'patched patchify diverges from original: {delta}'
    print(f'patchify replaced on {patched} modules, '
          f'self-check delta={delta:.2e}')
    onnx_path = out / 'model.onnx'
    torch.onnx.export(w, (sample,), onnx_path,
                      input_names=['context'],
                      output_names=['forecast'],
                      opset_version=17, dynamo=False)
    meta = {'context_length': ctx_len, 'horizon_steps': horizon,
            'source_dataset': 'uci-appliances-energy'
            if args.epochs else 'zero-shot'}
    return _parity(tout, onnx_path,
                   {'context': sample.numpy()}, args.atol), meta


# ── chronos-2 fine-tune (Kaggle GPU recipe) ──────────────────────────

def _train_chronos(args, out: Path) -> tuple[float, dict]:
    """Chronos2Pipeline.fit() LoRA on public energy data → separate
    catalog entry (zero-shot stays default). GPU intended — runs CPU
    but bound num_steps tightly."""
    import pandas as pd
    import torch
    from chronos import Chronos2Pipeline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'chronos fit on {device}')
    pipe = Chronos2Pipeline.from_pretrained(args.model,
                                          device_map=device)
    df = pd.read_csv(uci_appliances_energy())
    target = torch.tensor(df['Appliances'].values,
                          dtype=torch.float32).numpy()
    # window the series — feeding all ~20k points as one sequence OOMs
    # even a T4 (activations scale with length); many short windows is
    # also better LoRA data.
    win = args.fit_window
    inputs = [target[i:i + win] for i in
              range(0, len(target) - win, win)]
    print(f'fit inputs: {len(inputs)} windows of {win}')
    pipe = pipe.fit(inputs=inputs,
                    prediction_length=128,
                    context_length=512,   # match the artifact's context
                    finetune_mode='lora',
                    num_steps=args.steps,
                    batch_size=args.fit_batch,
                    output_dir=str(out / 'fit'))
    # persist the finetuned weights, then reuse the verified export
    # recipe — _export_chronos takes any path from_pretrained accepts
    fit_dir = out / 'fit' / 'finetuned'
    fit_dir.mkdir(parents=True, exist_ok=True)
    pipe.model.save_pretrained(fit_dir)
    from export import _export_chronos
    return _export_chronos(str(fit_dir), out, args.atol), \
        {'source_dataset': 'uci-appliances-energy',
         'finetune_mode': 'lora', 'num_steps': args.steps,
         'context_length': 512, 'horizon_steps': 128}


def _train_battery(args, out: Path) -> tuple[float, dict]:
    """battery-health: NASA PCoE Li-ion capacity-fade → SOH forecaster.

    Single-stage design (fits the one-ONNX manifest contract — the
    BP198 two-stage composite would need a new artifact type): a small
    MLP takes the last W normalised capacity readings and predicts SOH
    H cycles ahead, letting the appliance project end-of-life.
    """
    import torch
    from torch import nn

    cells = nasa_liion_capacities()
    if len(cells) < 2:
        raise SystemExit(f'too few cells parsed: {sorted(cells)}')

    W, H = 20, 4          # window length, forecast horizon (cycles)
    xs, ys = [], []
    for cell, caps in cells.items():
        soh = caps / caps[0]
        for i in range(len(soh) - W - H + 1):
            xs.append(soh[i:i + W])
            ys.append(soh[i + W + H - 1])
    X = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)[:, None]
    print(f'{len(X)} windows from {len(cells)} cells')

    net = nn.Sequential(nn.Linear(W, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(X), generator=g)
    split = int(len(X) * 0.9)
    tr_idx, te_idx = perm[:split], perm[split:]
    net.train()
    for ep in range(args.epochs):
        p = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        ep_loss = nb = 0
        for i in range(0, len(p), 64):
            idx = p[i:i + 64]
            loss = lossf(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        print(f'epoch {ep + 1}/{args.epochs} '
              f'loss={ep_loss / max(nb, 1):.5f}')
    net.eval()
    with torch.no_grad():
        mae = float(abs(net(X[te_idx]) - y[te_idx]).mean())
    print(f'held-out SOH MAE={mae:.4f}')

    sample = X[:1]
    with torch.no_grad():
        tout = net(sample).numpy()
    onnx_path = out / 'model.onnx'
    torch.onnx.export(net, (sample,), onnx_path,
                      input_names=['capacity'],
                      output_names=['soh'],
                      dynamic_axes={'capacity': {0: 'batch'},
                                    'soh': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    meta = {'cells': sorted(cells), 'window': W,
            'horizon_cycles': H, 'heldout_soh_mae': round(mae, 4),
            'source_dataset': 'nasa-pcoe-liion'}
    return _parity(tout, onnx_path,
                   {'capacity': sample.numpy()}, args.atol), meta


TRAIN_RECIPE = {'pq': _train_pq, 'anomaly': _train_anomaly,
                'ttm': _train_ttm, 'chronos': _train_chronos,
                'battery': _train_battery}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', required=True,
                    choices=sorted(TRAIN_RECIPE))
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', default='',
                    help='HF repo for fine-tune recipes')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--steps', type=int, default=500,
                    help='chronos fit() num_steps')
    ap.add_argument('--fit-window', type=int, default=2048,
                    help='chronos fit(): max points per input series')
    ap.add_argument('--fit-batch', type=int, default=8,
                    help='chronos fit() batch_size (256 default OOMs)')
    ap.add_argument('--samples-per-class', type=int, default=400)
    ap.add_argument('--windows', type=int, default=512)
    ap.add_argument('--channels', type=int, default=1)
    ap.add_argument('--window', type=int, default=1280)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--atol', type=float, default=1e-4)
    ap.add_argument('--title', default='')
    ap.add_argument('--description', default='')
    ap.add_argument('--use-case', default='')
    ap.add_argument('--tier', default='ai')
    ap.add_argument('--kind', default='')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    diff, meta = TRAIN_RECIPE[args.recipe](args, args.out)
    write_fragment(args.out, name=args.name,
                   source=args.model or f'trained:{args.recipe}',
                   diff=diff, meta=meta, title=args.title,
                   description=args.description, use_case=args.use_case,
                   tier=args.tier, kind=args.kind)


if __name__ == '__main__':
    main()
