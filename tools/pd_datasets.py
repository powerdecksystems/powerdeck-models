"""datasets.py — public + synthetic training data for generic recipes
(BP198 §5). HARD RULE: no customer data ever enters this module.
Everything here is either synthesized in-process or fetched from a
public dataset URL.

  ieee1159_waveforms()  — parametric PQ event generator (pq recipe)
  synthetic_telemetry() — normal-operation multichannel energy signals
                          (anomaly recipe)
  uci_appliances_energy() — UCI energydata_complete (ttm/chronos)
"""
from pathlib import Path

import numpy as np

UCI_APPLIANCES_CSV = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases'
    '/00374/energydata_complete.csv')
UCI_HOUSEHOLD_ZIP = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases'
    '/00235/household_power_consumption.zip')

PQ_CLASSES = [
    'normal', 'sag', 'swell', 'interruption', 'impulsive_transient',
    'oscillatory_transient', 'harmonics', 'flicker',
]


def ieee1159_waveforms(n_per_class: int = 400, window: int = 1280,
                      freq: float = 50.0, seed: int = 7):
    """IEEE-1159 event classes — each generator takes a clean window and
    applies its parametric disturbance. window default = 10 cycles at
    128 samples/cycle. Returns (X float32 [N,1,window], y int64)."""
    rng = np.random.default_rng(seed)
    t = np.arange(window, dtype=np.float32) * freq / (window / 10)
    # t counts cycles — clean 50/60 Hz carrier
    base = np.sin(2 * np.pi * t)

    def clean():
        return (base + rng.normal(0, 0.02, window)).astype(np.float32)

    def sag():
        x = clean()
        s = rng.integers(0, window // 2)
        dur = rng.integers(32, window // 2)
        x[s:s + dur] *= rng.uniform(0.1, 0.9)
        return x

    def swell():
        x = clean()
        s = rng.integers(0, window // 2)
        dur = rng.integers(32, window // 2)
        x[s:s + dur] *= rng.uniform(1.1, 1.8)
        return x

    def interruption():
        x = clean()
        s = rng.integers(0, window // 4)
        dur = rng.integers(64, window // 2)
        x[s:s + dur] *= rng.uniform(0.0, 0.1)
        return x

    def impulsive():
        x = clean()
        s = rng.integers(0, window - 8)
        x[s:s + rng.integers(2, 8)] += rng.uniform(3, 8) * rng.choice(
            [-1, 1])
        return x

    def oscillatory():
        x = clean()
        s = rng.integers(0, window // 2)
        dur = rng.integers(64, 256)
        f_hf = rng.uniform(8, 40)  # HF ringing multiple of carrier
        decay = np.exp(-np.arange(dur) / rng.uniform(20, 80))
        x[s:s + dur] += (rng.uniform(0.5, 1.5)
                         * np.sin(2 * np.pi * f_hf
                                  * np.arange(dur) / dur)
                         * decay)
        return x

    def harmonics():
        h3 = rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * 3 * t
                                             + rng.uniform(0, 6.28))
        h5 = rng.uniform(0.02, 0.2) * np.sin(2 * np.pi * 5 * t
                                             + rng.uniform(0, 6.28))
        h7 = rng.uniform(0.0, 0.1) * np.sin(2 * np.pi * 7 * t
                                            + rng.uniform(0, 6.28))
        return (base + h3 + h5 + h7
                + rng.normal(0, 0.02, window)).astype(np.float32)

    def flicker():
        fm = rng.uniform(0.5, 35)  # flicker frequency Hz equivalent
        depth = rng.uniform(0.02, 0.1)
        mod = 1 + depth * np.sin(2 * np.pi * fm
                                 * np.arange(window) / window
                                 + rng.uniform(0, 6.28))
        return (base * mod + rng.normal(0, 0.02, window)).astype(
            np.float32)

    gens = {
        'normal': clean, 'sag': sag, 'swell': swell,
        'interruption': interruption,
        'impulsive_transient': impulsive,
        'oscillatory_transient': oscillatory,
        'harmonics': harmonics, 'flicker': flicker,
    }
    xs, ys = [], []
    for idx, name in enumerate(PQ_CLASSES):
        for _ in range(n_per_class):
            xs.append(gens[name]())
            ys.append(idx)
    perm = rng.permutation(len(xs))
    X = np.stack(xs)[perm][:, None, :]
    return X.astype(np.float32), np.array(ys, dtype=np.int64)[perm]


def synthetic_telemetry(n_windows: int = 512, channels: int = 1,
                        length: int = 512, seed: int = 11):
    """Normal-operation energy telemetry — multi-frequency daily-cycle
    sinusoids + trend + noise. Mirrors the generator in the
    anomaly-transformer repo's train.py (base_signal) so self-supervised
    training sees the same distribution it was designed for.
    Returns float32 [n_windows, channels, length] — all normal."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_windows, channels, length), dtype=np.float32)
    for i in range(n_windows):
        for c in range(channels):
            tt = np.arange(length, dtype=np.float32)
            f = rng.uniform(0.005, 0.03)
            sig = (np.sin(2 * np.pi * f * tt)
                   + 0.5 * np.sin(2 * np.pi * f * 2.3 * tt + 0.7)
                   + 0.3 * np.sin(2 * np.pi * f * 5.1 * tt + 1.2)
                   + rng.normal(0, 0.05, length))
            out[i, c] = sig.astype(np.float32)
    return out


def uci_appliances_energy(cache: Path = Path('/tmp/uci-energy')):
    """UCI Appliances Energy Prediction (energydata_complete.csv,
    19,735 rows × 29 cols, 10-min intervals). Public dataset — the
    generic energy fine-tune corpus."""
    import urllib.request
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / 'energydata_complete.csv'
    if not dest.exists():
        urllib.request.urlretrieve(UCI_APPLIANCES_CSV, dest)
    return dest


NASA_BATTERY_ZIP = (
    'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip')


def nasa_liion_capacities(cells=('B0005', 'B0006', 'B0007', 'B0018',
                                 'B0029', 'B0030', 'B0031', 'B0032'),
                          cache: Path = Path('/tmp/nasa-liion')):
    """NASA PCoE Li-ion aging dataset (Saha & Goebel 2007) — public zip
    of .mat files; each cell's discharge-cycle capacity series.
    Returns dict {cell_id: float32 array of capacities (Ah per cycle)}.

    The outer zip contains nested zips; members are decompressed in
    memory and parsed with scipy.io.loadmat."""
    import io
    import urllib.request
    import zipfile
    cache.mkdir(parents=True, exist_ok=True)
    outer = cache / 'nasa-battery.zip'
    if not outer.exists():
        print(f'downloading {NASA_BATTERY_ZIP}')
        urllib.request.urlretrieve(NASA_BATTERY_ZIP, outer)
    import scipy.io
    wanted = {c.upper() for c in cells}
    members = []
    with zipfile.ZipFile(outer) as z:
        for n in z.namelist():
            if n.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(z.read(n))) as inner:
                    members.extend(inner.open(m).read()
                                   for m in inner.namelist()
                                   if Path(m).stem.upper() in wanted)
            elif Path(n).stem.upper() in wanted:
                members.append(z.read(n))
    # members are raw .mat bytes — keep only requested cells
    out = {}
    for blob in members:
        m = scipy.io.loadmat(io.BytesIO(blob))
        cell = next((k for k in m if k.upper() in wanted), None)
        if cell is None or cell in out:
            continue
        cycles = m[cell][0, 0]['cycle'][0]
        caps = []
        for cyc in cycles:
            if str(cyc['type'][0]) != 'discharge':
                continue
            try:
                caps.append(float(cyc['data'][0, 0]['Capacity'][0, 0]))
            except (IndexError, KeyError, ValueError):
                continue
        out[cell] = np.array(caps, dtype=np.float32)
        print(f'{cell}: {len(caps)} discharge cycles')
    missing = wanted - set(out)
    if missing:
        print(f'warning: cells not found in archive: {sorted(missing)}')
    return out
