"""
GenAI usage statement:
I used ChatGPT in an assistive way while developing this file. It helped mainly with
code structure, wording of comments/docstrings, and checking that the training output
was clearly organised. The final implementation decisions, experiment setup, MixUp and
label smoothing logic, early stopping behaviour, and verification of correctness were
done by me. I also checked that the script follows the coursework restrictions, runs on
CPU, and produces the required saved artifacts.

Task 2 training script:
- downloads and loads Fashion-MNIST,
- trains Task 2 ablations for none / MixUp / label smoothing / MixUp+label smoothing,
- applies manual early stopping,
- runs the controlled multi-seed protocol,
- saves checkpoints and ablation_history.json.
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

MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "35"))
PATIENCE = int(os.environ.get("PATIENCE", "6"))
BATCH_TRAIN = int(os.environ.get("BATCH_TRAIN", "128"))
BATCH_VAL = int(os.environ.get("BATCH_VAL", "256"))
LR = float(os.environ.get("LR", "0.06"))
MOMENTUM = float(os.environ.get("MOMENTUM", "0.9"))

TRAIN_N = int(os.environ.get("TRAIN_N", "15000"))
VAL_N = int(os.environ.get("VAL_N", "5000"))
SPLIT_SEED = 777

ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.4"))
LS_EPS = float(os.environ.get("LS_EPS", "0.10"))

WEIGHT_DECAY = 0.0  # keep Task2 focused on MixUp/LS

SEEDS = [0, 1, 2]
RUN_ALL_ABLATIONS_ALL_SEEDS = int(os.environ.get("RUN_ALL_ABLATIONS_ALL_SEEDS", "0"))


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible CPU-based experiments.

    Args:
        seed (int):
            Integer random seed used for Python's random module and PyTorch so that
            parameter initialisation, data-order effects, and other stochastic parts
            of training are more reproducible across runs.

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
    Fully connected neural network used for Task 2 classification on Fashion-MNIST.

    Purpose:
        Defines the multilayer perceptron architecture used in the Task 2 ablation,
        early-stopping, and multi-seed training experiments.

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

    def __init__(self, dropout_p: float = 0.10):
        """
        Initialise the Task 2 MLP model.

        Args:
            dropout_p (float):
                Dropout probability applied after selected hidden layers.
                Scalar in the range [0.0, 1.0]. In this task it is kept small and fixed
                across ablations so that the main comparison focuses on MixUp and label
                smoothing.

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


def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer class labels to one-hot floating-point targets.

    Args:
        y (torch.Tensor):
            Tensor of integer class labels with shape (batch_size,).

        num_classes (int):
            Total number of classes. For Fashion-MNIST this is 10.

    Returns:
        torch.Tensor:
            One-hot target tensor of shape (batch_size, num_classes) with float values.
    """
    return F.one_hot(y, num_classes=num_classes).float()


def mixup(x: torch.Tensor, y_soft: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp to one batch of inputs and soft targets.

    Args:
        x (torch.Tensor):
            Input image batch of shape (batch_size, 1, 28, 28).

        y_soft (torch.Tensor):
            Soft target tensor of shape (batch_size, num_classes), typically produced
            from one-hot labels and optionally later modified by label smoothing.

        alpha (float):
            Beta-distribution parameter controlling interpolation strength.
            Scalar value greater than 0. If alpha <= 0, the original batch is returned.

    Returns:
        tuple[torch.Tensor, torch.Tensor, float]:
            (x_mix, y_mix, lam), where:
            - x_mix is the mixed image batch of shape (batch_size, 1, 28, 28),
            - y_mix is the mixed soft-target tensor of shape (batch_size, num_classes),
            - lam is the sampled interpolation weight as a float.
    """
    if alpha <= 0:
        return x, y_soft, 1.0
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y_soft[idx]
    x_mix = lam * x + (1.0 - lam) * x2
    y_mix = lam * y_soft + (1.0 - lam) * y2
    return x_mix, y_mix, lam


def label_smooth(y_soft: torch.Tensor, eps: float, num_classes: int) -> torch.Tensor:
    """
    Apply label smoothing to a batch of soft targets.

    Args:
        y_soft (torch.Tensor):
            Soft target tensor of shape (batch_size, num_classes).

        eps (float):
            Smoothing strength. A value of 0.0 leaves the targets unchanged.

        num_classes (int):
            Total number of classes over which the smoothing mass is distributed.

    Returns:
        torch.Tensor:
            Smoothed soft-target tensor of shape (batch_size, num_classes).
    """
    if eps <= 0:
        return y_soft
    return (1.0 - eps) * y_soft + (eps / num_classes)


def soft_ce(logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    """
    Compute soft cross-entropy loss for logits and soft targets.

    Args:
        logits (torch.Tensor):
            Model output tensor of shape (batch_size, num_classes) containing
            unnormalised class scores.

        y_soft (torch.Tensor):
            Soft target tensor of shape (batch_size, num_classes).

    Returns:
        torch.Tensor:
            Scalar loss tensor equal to the mean soft cross-entropy over the batch.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(y_soft * log_probs).sum(dim=1).mean()


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Compute classification accuracy for a model on a dataset loader.

    Args:
        model (nn.Module):
            PyTorch classification model used for inference.

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


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device, cfg: dict) -> float:
    """
    Train the model for one epoch under a specified Task 2 configuration.

    Args:
        model (nn.Module):
            PyTorch model to be updated.

        loader (DataLoader):
            Training DataLoader yielding batches of the form (x, y), where:
            - x has shape (batch_size, 1, 28, 28),
            - y has shape (batch_size,).

        opt (torch.optim.Optimizer):
            Optimizer used to update the model parameters.

        device (torch.device):
            Device on which training is performed. In this script this is expected
            to be CPU.

        cfg (dict):
            Configuration dictionary controlling the current ablation run. It contains
            flags and hyperparameters such as:
            - cfg["mixup"]: whether MixUp is enabled,
            - cfg["ls"]: whether label smoothing is enabled,
            - cfg["alpha"]: MixUp alpha value,
            - cfg["eps"]: label smoothing epsilon.

    Returns:
        float:
            Average soft cross-entropy training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_soft = one_hot(y, 10)

        if cfg["mixup"]:
            x, y_soft, _ = mixup(x, y_soft, cfg["alpha"])
        if cfg["ls"]:
            y_soft = label_smooth(y_soft, cfg["eps"], 10)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = soft_ce(logits, y_soft)
        loss.backward()
        opt.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(1, total)


def train_cfg(seed: int, cfg: dict, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, out_dir: Path) -> dict:
    """
    Train one Task 2 ablation configuration with manual early stopping.

    Purpose:
        Runs a complete training experiment for one seed and one configuration,
        records validation history, applies patience-based early stopping, and
        saves the best checkpoint for that run.

    Args:
        seed (int):
            Random seed used for this run.

        cfg (dict):
            Configuration dictionary describing the ablation condition and checkpoint
            name for the run.

        train_loader (DataLoader):
            Training DataLoader yielding input-image and label batches.

        val_loader (DataLoader):
            Validation DataLoader yielding input-image and label batches.

        device (torch.device):
            Device used for training and evaluation.

        out_dir (Path):
            Output directory in which the best checkpoint is saved.

    Returns:
        dict:
            Run record containing:
            - best_val,
            - best_epoch,
            - epochs_ran,
            - history,
            - config,
            together with identifying metadata for the run.
    """
    set_seed(seed)
    model = DeepMLP(0.10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    best_val = -1.0
    best_epoch = 0
    patience_left = PATIENCE
    hist = {"train_loss": [], "val_acc": []}

    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device, cfg)
        va = accuracy(model, val_loader, device)
        dt = time.time() - t0

        hist["train_loss"].append(float(tr_loss))
        hist["val_acc"].append(float(va))

        if va > best_val + 1e-4:
            best_val = va
            best_epoch = ep
            patience_left = PATIENCE
            torch.save(model.state_dict(), out_dir / cfg["ckpt"])
        else:
            patience_left -= 1

        print(
            f"[seed={seed} {cfg['name']:<8}] ep {ep:02d}/{MAX_EPOCHS} | "
            f"loss {tr_loss:.4f} | val {va*100:.2f}% | best {best_val*100:.2f}% @ep{best_epoch} | patience {patience_left} | {dt:.1f}s"
        )

        if patience_left <= 0:
            break

    return {
        "seed": int(seed),
        "name": cfg["name"],
        "ckpt": cfg["ckpt"],
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(hist["val_acc"])),
        "history": hist,
        "config": cfg,
    }


def main() -> None:
    """
    Execute the full Task 2 training pipeline.

    Purpose:
        Downloads and loads Fashion-MNIST, creates the fixed train/validation split,
        trains the Task 2 ablation models, applies manual early stopping, runs the
        controlled multi-seed protocol, and saves checkpoints together with
        ablation_history.json.

    Inputs:
        None directly from function arguments.
        The function uses configuration values defined near the top of the file,
        including epoch limit, patience, batch sizes, learning rate, momentum,
        dataset sizes, MixUp alpha, label smoothing epsilon, and selected seeds.

    Returns:
        None.
            The function writes checkpoints and ablation_history.json to the current
            task directory and prints progress summaries to the terminal.
    """
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    tfm = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=tfm)

    subset = Subset(full_train, list(range(TRAIN_N + VAL_N)))
    train_ds, val_ds = random_split(subset, [TRAIN_N, VAL_N], generator=torch.Generator().manual_seed(SPLIT_SEED))

    results = []
    t_all = time.time()

    base_cfgs = [
        {"name": "none",     "mixup": False, "ls": False, "alpha": 0.0,   "eps": 0.0},
        {"name": "mixup",    "mixup": True,  "ls": False, "alpha": ALPHA, "eps": 0.0},
        {"name": "ls",       "mixup": False, "ls": True,  "alpha": 0.0,   "eps": LS_EPS},
        {"name": "mixup+ls", "mixup": True,  "ls": True,  "alpha": ALPHA, "eps": LS_EPS},
    ]

    print("\n=== Task 2 training (ablations + multi-seed reliability) ===")
    print(f"train={TRAIN_N}, val={VAL_N}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}, lr={LR}, momentum={MOMENTUM}, wd={WEIGHT_DECAY}")
    print(f"mixup alpha={ALPHA}, label smoothing eps={LS_EPS}")
    print(f"seeds={SEEDS}, RUN_ALL_ABLATIONS_ALL_SEEDS={RUN_ALL_ABLATIONS_ALL_SEEDS}\n")

    for seed in SEEDS:
        # tie shuffle to seed
        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False, num_workers=0)

        # runtime-controlled: seed0 trains all configs; other seeds train only main unless flag is set
        cfgs_to_run = base_cfgs if (seed == 0 or RUN_ALL_ABLATIONS_ALL_SEEDS == 1) else [base_cfgs[-1]]

        for cfg in cfgs_to_run:
            # checkpoint naming:
            # keep seed0 "mixup_ls_best.pth" compatibility, but also save seed-specific names for reliability
            if cfg["name"] == "mixup+ls":
                ckpt = "mixup_ls_best.pth" if seed == 0 else f"mixup_ls_seed{seed}_best.pth"
            else:
                ckpt = f"{cfg['name']}_best.pth" if seed == 0 else f"{cfg['name']}_seed{seed}_best.pth"

            cfg_run = dict(cfg)
            cfg_run["ckpt"] = ckpt

            results.append(train_cfg(seed, cfg_run, train_loader, val_loader, device, here))
            print("")

    out = {
        "meta": {
            "dataset": "FashionMNIST",
            "train_size": TRAIN_N,
            "val_size": VAL_N,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_train": BATCH_TRAIN,
            "batch_val": BATCH_VAL,
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "mixup_alpha": ALPHA,
            "ls_eps": LS_EPS,
            "split_seed": SPLIT_SEED,
            "seeds": SEEDS,
            "run_all_ablations_all_seeds": RUN_ALL_ABLATIONS_ALL_SEEDS,
        },
        "results": results,
    }
    with open(here / "ablation_history.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    dt = time.time() - t_all
    print(f"Saved: ablation_history.json | total_time={dt:.1f}s")
    print("Saved checkpoints:")
    for r in results:
        print(" -", r["ckpt"])


if __name__ == "__main__":
    main()