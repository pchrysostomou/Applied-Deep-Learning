"""
GenAI usage statement:
I used ChatGPT in an assistive way while developing this file. It helped mainly with
code structure, wording of comments/docstrings, and checking that the training output
was clearly organised. The final implementation decisions, experiment design, selected
hyperparameters, and verification of correctness were done by me. I also checked that
the script follows the coursework restrictions, runs on CPU, and produces the required
saved artifacts.

Task 1 training script:
- downloads and loads Fashion-MNIST,
- runs controlled regularization ablations,
- runs multi-seed experiments,
- runs an optimizer momentum comparison,
- saves history.json, runs.json, model weights, and main aliases.
"""

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# -----------------------------
# CPU-friendly defaults (override via env vars)
# -----------------------------
EPOCHS = int(os.environ.get("EPOCHS", "20"))
TRAIN_N = int(os.environ.get("TRAIN_N", "10000"))
VAL_N = int(os.environ.get("VAL_N", "10000"))
BATCH_TRAIN = int(os.environ.get("BATCH_TRAIN", "128"))
BATCH_VAL = int(os.environ.get("BATCH_VAL", "256"))
LR = float(os.environ.get("LR", "0.08"))
DEFAULT_MOMENTUM = float(os.environ.get("MOMENTUM", "0.9"))

SEEDS = [0, 1, 2]
SPLIT_SEED = 12345  # fixed split across seeds (fair comparison)

CONDITIONS = {
    "baseline": {"dropout": 0.0, "weight_decay": 0.0},
    "dropout_only": {"dropout": 0.30, "weight_decay": 0.0},
    "weight_decay_only": {"dropout": 0.0, "weight_decay": 5e-4},
    "dropout_weight_decay": {"dropout": 0.30, "weight_decay": 5e-4},
}

OPTIMIZER_ABLATIONS = [
    {"tag": "sgd_m0.0", "momentum": 0.0},
    {"tag": "sgd_m0.9", "momentum": 0.9},
]


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible CPU-based experiments.

    Args:
        seed (int):
            Integer random seed used for Python's random module and PyTorch so that
            training order, parameter initialisation, and other stochastic behaviour
            are more reproducible across runs.

    Returns:
        None.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepMLP(nn.Module):
    """
    Fully connected neural network used for Task 1 classification on Fashion-MNIST.

    Purpose:
        Defines the multilayer perceptron architecture used in the controlled training
        experiments for baseline, regularization ablations, and optimizer comparison.

    Input:
        The network expects image batches originally shaped as
        torch.Tensor of size (batch_size, 1, 28, 28), where:
        - batch_size is the number of grayscale images in the batch,
        - 1 is the channel count,
        - 28 x 28 is the Fashion-MNIST image resolution.

    Output:
        torch.Tensor of size (batch_size, 10) containing unnormalised class scores
        (logits) for the 10 Fashion-MNIST classes.
    """

    def __init__(self, dropout_p: float = 0.0):
        """
        Initialise the Task 1 MLP model.

        Args:
            dropout_p (float):
                Dropout probability applied after selected hidden layers.
                Scalar in the range [0.0, 1.0].
                A value of 0.0 means no dropout is applied.

        Returns:
            None.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the network.

        Args:
            x (torch.Tensor):
                Input image batch of shape (batch_size, 1, 28, 28), where each item is
                a grayscale Fashion-MNIST image. The tensor is flattened internally to
                shape (batch_size, 784) before being passed through the MLP.

        Returns:
            torch.Tensor:
                Output logits of shape (batch_size, 10), where each row contains the
                predicted class scores for one input image.
        """
        x = x.view(x.size(0), -1)
        return self.net(x)


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Compute classification accuracy for a model on a dataset loader.

    Args:
        model (nn.Module):
            PyTorch classification model that maps an input batch to class logits.

        loader (DataLoader):
            DataLoader yielding batches of the form (x, y), where:
            - x is a tensor of images with shape (batch_size, 1, 28, 28),
            - y is a tensor of integer class labels with shape (batch_size,).

        device (torch.device):
            Device on which inference is performed. In this coursework script this is
            expected to be CPU.

    Returns:
        float:
            Scalar accuracy in the range [0.0, 1.0].
    """
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """
    Train the model for one full epoch over the training loader.

    Args:
        model (nn.Module):
            PyTorch model to be updated.

        loader (DataLoader):
            DataLoader yielding training batches of the form (x, y), where:
            - x has shape (batch_size, 1, 28, 28),
            - y has shape (batch_size,).

        optimizer (torch.optim.Optimizer):
            Optimizer used to update the model parameters.

        device (torch.device):
            Device on which training is performed. In this script this is expected
            to be CPU.

    Returns:
        float:
            Average cross-entropy training loss over the epoch.
    """

    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(1, total)


def summarize_run(train_acc: list, val_acc: list) -> dict:
    """
    Build summary metrics for one experimental training run.

    Args:
        train_acc (list):
            One-dimensional list of per-epoch training accuracies as floats in the
            range [0.0, 1.0].

        val_acc (list):
            One-dimensional list of per-epoch validation accuracies as floats in the
            range [0.0, 1.0]. This list is expected to have the same length as
            train_acc.

    Returns:
        dict:
            Dictionary containing:
            - best_val: best validation accuracy,
            - best_epoch: epoch index (1-based) of best validation accuracy,
            - gap_at_best: train/validation accuracy gap at the best validation epoch,
            - final_gap: train/validation accuracy gap at the final epoch.
    """
    best_i = 0
    best_val = val_acc[0]
    for i, v in enumerate(val_acc):
        if v > best_val:
            best_val = v
            best_i = i

    return {
        "best_val": float(best_val),
        "best_epoch": int(best_i + 1),
        "gap_at_best": float(train_acc[best_i] - val_acc[best_i]),
        "final_gap": float(train_acc[-1] - val_acc[-1]),
    }


def train_run(
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seed: int,
    cond_name: str,
    dropout: float,
    weight_decay: float,
    momentum: float,
    out_dir: Path,
) -> dict:
    """
    Train one complete experimental run for a given condition, seed, and momentum.

    Purpose:
        Runs the model for the configured number of epochs, records loss/accuracy
        curves, computes summary metrics, and optionally saves seed-0 weights.

    Args:
        device (torch.device):
            Device used for training and evaluation.

        train_loader (DataLoader):
            Training DataLoader yielding batches of input images and labels.

        val_loader (DataLoader):
            Validation DataLoader yielding batches of input images and labels.

        seed (int):
            Random seed used for this run.

        cond_name (str):
            Condition label describing the experiment, for example 'baseline' or
            'dropout_weight_decay'.

        dropout (float):
            Dropout probability used in the model for this run.

        weight_decay (float):
            L2 weight decay coefficient used by SGD for this run.

        momentum (float):
            SGD momentum value used for this run.

        out_dir (Path):
            Output directory in which seed-0 model weights may be saved.

    Returns:
        dict:
            Run record containing metadata, full per-epoch curves, and summary
            statistics for the experiment.
    """
    set_seed(seed)

    model = DeepMLP(dropout_p=dropout).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)

    train_loss, train_acc, val_acc = [], [], []
    for _ in range(EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, opt, device)
        tr_acc = accuracy(model, train_loader, device)
        va_acc = accuracy(model, val_loader, device)
        train_loss.append(float(tr_loss))
        train_acc.append(float(tr_acc))
        val_acc.append(float(va_acc))

    summary = summarize_run(train_acc, val_acc)

    run = {
        "condition": cond_name,
        "seed": int(seed),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "momentum": float(momentum),
        "lr": float(LR),
        "epochs": int(EPOCHS),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "summary": summary,
    }

    if seed == 0:
        wname = f"{cond_name}_seed0_mom{momentum:.1f}.pth"
        torch.save(model.state_dict(), out_dir / wname)

    return run


def main() -> None:
    """
    Execute the full Task 1 training pipeline.

    Purpose:
        Downloads and loads Fashion-MNIST, creates the fixed train/validation split,
        runs the regularization ablations, runs the multi-seed experiments, runs the
        optimizer momentum comparison, and saves the required JSON and weight files.

    Inputs:
        None directly from function arguments.
        The function uses configuration values defined near the top of the file,
        including epoch count, dataset sizes, batch sizes, learning rate, momentum,
        and selected random seeds.

    Returns:
        None.
            The function writes output artifacts to the current task directory and
            prints progress summaries to the terminal.
    """
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    tfm = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=tfm)

    subset = Subset(full_train, list(range(TRAIN_N + VAL_N)))
    train_ds, val_ds = random_split(
        subset,
        [TRAIN_N, VAL_N],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )

    all_runs = []
    t_all = time.time()

    print("\n=== Task 1 training (ablations + multi-seed + optimizer comparison) ===")
    print(f"epochs={EPOCHS}, train={TRAIN_N}, val={VAL_N}, lr={LR}, momentum={DEFAULT_MOMENTUM}, seeds={SEEDS}\n")

    for seed in SEEDS:
        # shuffle tied to seed
        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False, num_workers=0)

        # (1) Conditions
        for cond_name, cfg in CONDITIONS.items():
            t0 = time.time()
            run = train_run(
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                seed=seed,
                cond_name=cond_name,
                dropout=cfg["dropout"],
                weight_decay=cfg["weight_decay"],
                momentum=DEFAULT_MOMENTUM,
                out_dir=here,
            )
            all_runs.append(run)
            dt = time.time() - t0
            s = run["summary"]
            print(
                f"[{cond_name:>18}] seed={seed} | "
                f"best_val={s['best_val']*100:.2f}% @ep{s['best_epoch']:02d} | "
                f"gap_best={s['gap_at_best']*100:.2f}% | final_gap={s['final_gap']*100:.2f}% | {dt:.1f}s"
            )

        # (2) Optimizer ablation for baseline: momentum=0.0 only (0.9 already baseline)
        for opt in OPTIMIZER_ABLATIONS:
            if abs(opt["momentum"] - DEFAULT_MOMENTUM) < 1e-9:
                continue
            t0 = time.time()
            run = train_run(
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                seed=seed,
                cond_name="baseline_optimizer_ablation",
                dropout=0.0,
                weight_decay=0.0,
                momentum=opt["momentum"],
                out_dir=here,
            )
            run["optimizer_tag"] = opt["tag"]
            all_runs.append(run)
            dt = time.time() - t0
            s = run["summary"]
            print(
                f"[{'baseline_opt':>18}] seed={seed} | mom={opt['momentum']:.1f} | "
                f"best_val={s['best_val']*100:.2f}% @ep{s['best_epoch']:02d} | "
                f"gap_best={s['gap_at_best']*100:.2f}% | final_gap={s['final_gap']*100:.2f}% | {dt:.1f}s"
            )

        print("")

    def pick(cond: str, seed: int, momentum: float) -> dict | None:
        """
        Retrieve one specific run record from the list of completed runs.

        Args:
            cond (str):
                Experiment condition name to match.

            seed (int):
                Random seed value to match.

            momentum (float):
                Momentum value to match.

        Returns:
            dict | None:
                Matching run dictionary if found, otherwise None.
        """
        for r in all_runs:
            if r["condition"] == cond and r["seed"] == seed and abs(r["momentum"] - momentum) < 1e-9:
                return r
        return None

    seed0_baseline = pick("baseline", 0, DEFAULT_MOMENTUM)
    seed0_reg = pick("dropout_weight_decay", 0, DEFAULT_MOMENTUM)
    if seed0_baseline is None or seed0_reg is None:
        raise RuntimeError("Missing seed0 baseline/regularized runs (unexpected).")

    # history.json for required plot (seed0 baseline vs seed0 regularized)
    history = {
        "meta": {
            "dataset": "FashionMNIST",
            "train_size": TRAIN_N,
            "val_size": VAL_N,
            "batch_size": BATCH_TRAIN,
            "epochs": EPOCHS,
            "optimizer": f"SGD(momentum={DEFAULT_MOMENTUM})",
            "lr": LR,
            "seed_for_main_plot": 0,
            "baseline": CONDITIONS["baseline"],
            "regularized": CONDITIONS["dropout_weight_decay"],
            "note": "history.json includes only seed0 baseline vs dropout+weight_decay for required plot. Full ablations are in runs.json.",
        },
        "baseline": {"train_acc": seed0_baseline["train_acc"], "val_acc": seed0_baseline["val_acc"], "train_loss": seed0_baseline["train_loss"]},
        "regularized": {"train_acc": seed0_reg["train_acc"], "val_acc": seed0_reg["val_acc"], "train_loss": seed0_reg["train_loss"]},
    }
    with open(here / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # runs.json for all evidence
    runs_out = {
        "meta": {
            "epochs": EPOCHS,
            "train_size": TRAIN_N,
            "val_size": VAL_N,
            "batch_train": BATCH_TRAIN,
            "batch_val": BATCH_VAL,
            "lr": LR,
            "default_momentum": DEFAULT_MOMENTUM,
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "optimizer_ablation": OPTIMIZER_ABLATIONS,
            "split_seed": SPLIT_SEED,
        },
        "runs": all_runs,
    }
    with open(here / "runs.json", "w", encoding="utf-8") as f:
        json.dump(runs_out, f, indent=2)

    # required aliases
    b_src = here / f"baseline_seed0_mom{DEFAULT_MOMENTUM:.1f}.pth"
    r_src = here / f"dropout_weight_decay_seed0_mom{DEFAULT_MOMENTUM:.1f}.pth"
    if b_src.exists():
        torch.save(torch.load(b_src, map_location="cpu"), here / "baseline.pth")
    if r_src.exists():
        torch.save(torch.load(r_src, map_location="cpu"), here / "regularized.pth")

    dt = time.time() - t_all
    print(f"\nDone. Total time: {dt:.1f}s")
    print("Saved: history.json, runs.json, baseline.pth, regularized.pth (+ seed0 ablation weights)")


if __name__ == "__main__":
    main()