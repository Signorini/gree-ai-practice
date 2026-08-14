"""
Green AI: Efficiency-Accuracy Trade-offs in Image Classification via Model Compression
=====================================================================================

Reproducible experiment for the Sustainable AI TABA.

Pipeline
--------
1. Train a baseline convolutional neural network (the "teacher") on CIFAR-10.
2. Derive three compressed variants:
     (a) Pruning       - global L1 unstructured pruning + short fine-tune.
     (b) Quantisation  - post-training dynamic INT8 quantisation.
     (c) Distillation  - a much smaller "student" trained with knowledge distillation.
3. For every model, measure a battery of *sustainability* and *quality* metrics:
     - Test accuracy (%)
     - Trainable / non-zero parameters
     - Model size on disk (MB)
     - Multiply-accumulate operations (MACs / "FLOPs") via thop
     - Inference latency (ms/image, CPU)
     - Training energy (kWh) and CO2e (kg) via CodeCarbon
     - Inference energy (kWh) and CO2e (kg) via CodeCarbon

All raw numbers are written to results/metrics.json and results/results_table.csv,
and two trade-off figures are saved under results/.

The configuration is intentionally lightweight so the whole study runs on a CPU in
a few minutes; scale EPOCHS / TRAIN_SUBSET up for publication-grade numbers.

"""

from __future__ import annotations

import copy
import io
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------


@dataclass
class Config:
    seed: int = 42
    data_dir: str = "./data"
    train_parquet: str = "./data/cifar10_train.parquet"
    test_parquet: str = "./data/cifar10_test.parquet"
    results_dir: str = "./results"
    country_iso_code: str = "IRL"          # for CodeCarbon offline grid intensity
    batch_size: int = 128
    epochs_teacher: int = 6
    epochs_student: int = 6
    epochs_prune_finetune: int = 3
    lr: float = 1e-3
    train_subset: int = 0                   # 0 = use full 50k train set
    test_subset: int = 0                    # 0 = use full 10k test set
    prune_amount: float = 0.5               # global sparsity target for pruning
    kd_temperature: float = 4.0
    kd_alpha: float = 0.7                   # weight on soft (teacher) loss
    latency_batches: int = 20               # batches timed for latency estimate
    num_workers: int = 0                    # data is in RAM; workers add spawn overhead on macOS


CFG = Config()

CIFAR10_CLASSES = 10


# ----------------------------------------------------------------------------------
# Reproducibility & environment
# ----------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # This study targets CPU deployment (the common "edge"/green scenario) so we
    # keep everything on CPU for a fair, comparable energy measurement.
    return torch.device("cpu")


# ----------------------------------------------------------------------------------
# CodeCarbon helper (graceful degradation if it cannot read power)
# ----------------------------------------------------------------------------------


def make_tracker(project_name: str):
    """Return an OfflineEmissionsTracker context-like object or a no-op fallback."""
    try:
        from codecarbon import OfflineEmissionsTracker

        return OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code=CFG.country_iso_code,
            output_dir=CFG.results_dir,
            log_level="error",
            measure_power_secs=1,
            save_to_file=False,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[codecarbon] disabled ({exc}); energy will be reported as 0.")

        class _NullTracker:
            def start(self):
                return None

            def stop(self):
                return 0.0

        return _NullTracker()


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------


class ParquetCIFAR(Dataset):
    """CIFAR-10 loaded from a HuggingFace parquet file (img=PNG bytes, label=int).

    PNGs are decoded once into an in-memory uint8 array for fast epochs.
    """

    def __init__(self, parquet_path: str, transform=None):
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        imgs = table.column("img").to_pylist()
        labels = table.column("label").to_pylist()
        arr = np.empty((len(imgs), 32, 32, 3), dtype=np.uint8)
        for i, rec in enumerate(imgs):
            with Image.open(io.BytesIO(rec["bytes"])) as im:
                arr[i] = np.asarray(im.convert("RGB"))
        self.data = arr
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_loaders(cfg: Config):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = ParquetCIFAR(cfg.train_parquet, transform=train_tf)
    test_set = ParquetCIFAR(cfg.test_parquet, transform=test_tf)

    if cfg.train_subset > 0:
        idx = np.random.choice(len(train_set), cfg.train_subset, replace=False)
        train_set = Subset(train_set, idx.tolist())
    if cfg.test_subset > 0:
        idx = np.random.choice(len(test_set), cfg.test_subset, replace=False)
        test_set = Subset(test_set, idx.tolist())

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False,
    )
    return train_loader, test_loader


# ----------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------


class TeacherCNN(nn.Module):
    """A compact VGG-style baseline (the 'non-green' reference model)."""

    def __init__(self, num_classes: int = CIFAR10_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                             # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class StudentCNN(nn.Module):
    """A much smaller network trained via knowledge distillation."""

    def __init__(self, num_classes: int = CIFAR10_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 16 -> 8
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------------


def train_supervised(model, loader, device, epochs, lr, tag=""):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running, seen, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            correct += (out.argmax(1) == y).sum().item()
        sched.step()
        print(f"  [{tag}] epoch {epoch + 1}/{epochs} "
              f"loss={running / seen:.3f} acc={100 * correct / seen:.2f}%")
    return model


def train_distillation(student, teacher, loader, device, epochs, lr, temp, alpha, tag="KD"):
    student.to(device).train()
    teacher.to(device).eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    for epoch in range(epochs):
        running, seen, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            opt.zero_grad()
            s_logits = student(x)
            soft = kl(
                F.log_softmax(s_logits / temp, dim=1),
                F.softmax(t_logits / temp, dim=1),
            ) * (temp * temp)
            hard = ce(s_logits, y)
            loss = alpha * soft + (1 - alpha) * hard
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            correct += (s_logits.argmax(1) == y).sum().item()
        sched.step()
        print(f"  [{tag}] epoch {epoch + 1}/{epochs} "
              f"loss={running / seen:.3f} acc={100 * correct / seen:.2f}%")
    return student


@torch.no_grad()
def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ----------------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------------


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    nonzero = int(sum((p != 0).sum().item() for p in model.parameters()))
    return total, nonzero


def model_size_mb(model) -> float:
    path = os.path.join(CFG.results_dir, "_tmp_state.pt")
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size


def macs_flops(model) -> float:
    """Return MACs (multiply-accumulates) in millions for a single 32x32 image."""
    try:
        from thop import profile

        m = copy.deepcopy(model).eval()
        dummy = torch.randn(1, 3, 32, 32)
        macs, _ = profile(m, inputs=(dummy,), verbose=False)
        return macs / 1e6
    except Exception as exc:  # pragma: no cover
        print(f"[thop] MAC counting failed ({exc}).")
        return float("nan")


@torch.no_grad()
def measure_latency(model, loader, device, n_batches: int) -> float:
    model.to(device).eval()
    # warm-up
    it = iter(loader)
    x, _ = next(it)
    for _ in range(2):
        model(x.to(device))
    times, imgs = 0.0, 0
    it = iter(loader)
    for i, (x, _) in enumerate(it):
        if i >= n_batches:
            break
        x = x.to(device)
        t0 = time.perf_counter()
        model(x)
        times += time.perf_counter() - t0
        imgs += x.size(0)
    return 1000.0 * times / imgs  # ms per image


@torch.no_grad()
def measure_inference_energy(model, loader, device, tag):
    tracker = make_tracker(f"infer_{tag}")
    tracker.start()
    model.to(device).eval()
    for x, _ in loader:
        model(x.to(device))
    kwh = tracker.stop() or 0.0
    # CodeCarbon returns emissions (kg); energy is stored on the tracker.
    energy = getattr(getattr(tracker, "_total_energy", None), "kWh", None)
    energy_kwh = float(energy) if energy is not None else float("nan")
    return energy_kwh, float(kwh)


def collect_metrics(model, name, loader, device, train_energy_kwh, train_co2_kg):
    total, nonzero = count_params(model)
    acc = evaluate(model, loader, device)
    size = model_size_mb(model)
    macs = macs_flops(model)
    latency = measure_latency(model, loader, device, CFG.latency_batches)
    infer_kwh, infer_co2 = measure_inference_energy(model, loader, device, name)
    row = {
        "model": name,
        "accuracy_pct": round(acc, 2),
        "params_total": total,
        "params_nonzero": nonzero,
        "sparsity_pct": round(100.0 * (1 - nonzero / total), 2) if total else 0.0,
        "size_mb": round(size, 3),
        "macs_million": round(macs, 2) if macs == macs else None,
        "latency_ms_per_img": round(latency, 3),
        "train_energy_kwh": round(train_energy_kwh, 8),
        "train_co2_kg": round(train_co2_kg, 8),
        "infer_energy_kwh": round(infer_kwh, 8) if infer_kwh == infer_kwh else None,
        "infer_co2_kg": round(infer_co2, 8),
    }
    print(f"  -> {name}: acc={acc:.2f}% size={size:.2f}MB "
          f"MACs={macs:.1f}M lat={latency:.2f}ms")
    return row


# ----------------------------------------------------------------------------------
# Compression strategies
# ----------------------------------------------------------------------------------


def apply_pruning(model, loader, device, amount, finetune_epochs, lr):
    pruned = copy.deepcopy(model)
    params_to_prune = [
        (m, "weight") for m in pruned.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    # short fine-tune to recover accuracy (energy measured by caller)
    train_supervised(pruned, loader, device, finetune_epochs, lr, tag="Prune-FT")
    for m, n in params_to_prune:
        prune.remove(m, n)  # make sparsity permanent
    return pruned


def apply_dynamic_quantisation(model):
    # Select an available quantised backend (qnnpack on ARM/Apple Silicon,
    # fbgemm on x86). Without this, linear_prepack raises "NoQEngine".
    for engine in ("qnnpack", "fbgemm"):
        if engine in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = engine
            break
    m = copy.deepcopy(model).to("cpu").eval()
    q = torch.quantization.quantize_dynamic(
        m, {nn.Linear}, dtype=torch.qint8
    )
    return q


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------


def main():
    os.makedirs(CFG.results_dir, exist_ok=True)
    set_seed(CFG.seed)
    device = get_device()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Device: {device} | threads: {torch.get_num_threads()}")

    train_loader, test_loader = get_loaders(CFG)

    results = []

    # ---- 1. Baseline (teacher) --------------------------------------------------
    print("\n[1/4] Training baseline teacher CNN ...")
    teacher = TeacherCNN()
    tr = make_tracker("train_teacher")
    tr.start()
    train_supervised(teacher, train_loader, device, CFG.epochs_teacher, CFG.lr, tag="Teacher")
    t_co2 = tr.stop() or 0.0
    t_e = getattr(getattr(tr, "_total_energy", None), "kWh", 0.0)
    t_e = float(t_e) if t_e is not None else 0.0
    results.append(collect_metrics(teacher, "Baseline (Teacher)", test_loader, device, t_e, t_co2))

    # ---- 2. Pruning -------------------------------------------------------------
    print("\n[2/4] Applying magnitude pruning + fine-tune ...")
    pr = make_tracker("train_prune")
    pr.start()
    pruned = apply_pruning(teacher, train_loader, device, CFG.prune_amount,
                           CFG.epochs_prune_finetune, CFG.lr)
    p_co2 = pr.stop() or 0.0
    p_e = getattr(getattr(pr, "_total_energy", None), "kWh", 0.0)
    p_e = float(p_e) if p_e is not None else 0.0
    results.append(collect_metrics(pruned, "Pruned (50%)", test_loader, device, p_e, p_co2))

    # ---- 3. Quantisation --------------------------------------------------------
    print("\n[3/4] Post-training dynamic INT8 quantisation ...")
    quant = apply_dynamic_quantisation(teacher)
    # quantisation is training-free -> training energy = 0
    results.append(collect_metrics(quant, "Quantised (INT8 dyn.)", test_loader, device, 0.0, 0.0))

    # ---- 4. Knowledge distillation ---------------------------------------------
    print("\n[4/4] Training distilled student ...")
    student = StudentCNN()
    kd = make_tracker("train_student")
    kd.start()
    train_distillation(student, teacher, train_loader, device, CFG.epochs_student,
                       CFG.lr, CFG.kd_temperature, CFG.kd_alpha, tag="Student-KD")
    s_co2 = kd.stop() or 0.0
    s_e = getattr(getattr(kd, "_total_energy", None), "kWh", 0.0)
    s_e = float(s_e) if s_e is not None else 0.0
    results.append(collect_metrics(student, "Distilled Student", test_loader, device, s_e, s_co2))

    # ---- Persist ----------------------------------------------------------------
    out_json = os.path.join(CFG.results_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump({"config": asdict(CFG), "results": results}, f, indent=2)

    # CSV
    import csv
    keys = list(results[0].keys())
    with open(os.path.join(CFG.results_dir, "results_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    _plot(results)
    print(f"\nDone. Metrics written to {out_json}")
    _print_table(results)


def _print_table(results):
    cols = ["model", "accuracy_pct", "params_total", "size_mb",
            "macs_million", "latency_ms_per_img", "train_energy_kwh", "train_co2_kg"]
    header = " | ".join(f"{c:>18}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(" | ".join(f"{str(r.get(c, '')):>18}" for c in cols))


def _plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["model"] for r in results]
        acc = [r["accuracy_pct"] for r in results]
        size = [r["size_mb"] for r in results]
        macs = [r["macs_million"] or 0 for r in results]

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].bar(names, size, color="#2e7d32")
        ax[0].set_ylabel("Model size (MB)")
        ax[0].set_title("Model size by compression strategy")
        ax[0].tick_params(axis="x", rotation=25)

        ax[1].scatter(macs, acc, s=90, color="#1565c0")
        for n, m, a in zip(names, macs, acc):
            ax[1].annotate(n, (m, a), fontsize=8, xytext=(4, 4),
                           textcoords="offset points")
        ax[1].set_xlabel("MACs (millions) - compute per image")
        ax[1].set_ylabel("Test accuracy (%)")
        ax[1].set_title("Accuracy vs. compute trade-off")
        fig.tight_layout()
        fig.savefig(os.path.join(CFG.results_dir, "tradeoff.png"), dpi=130)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[plot] skipped ({exc}).")


if __name__ == "__main__":
    main()
