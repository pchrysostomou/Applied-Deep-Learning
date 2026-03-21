"""
GenAI usage statement:
I used ChatGPT in an assistive way while developing this file. It helped mainly with
code structure, wording of comments/docstrings, and checking that the printed analysis
was clearly organised. The final implementation decisions, evaluation protocol,
robustness sweep design, montage content, and verification of correctness were done by me.
I also checked that the script follows the coursework restrictions, runs on CPU, and
produces the required saved artifacts without matplotlib.

Task 2 evaluation script:
- loads Fashion-MNIST test data,
- evaluates clean and noisy accuracy for the Task 2 ablations,
- reports Multi-seed reliability analysis (not performed in this run due to unavailable
additional seed checkpoints) for MixUp+Label Smoothing,
- reports a robustness sweep across multiple noise levels,
- saves robustness_demo.png with annotated MixUp evidence.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

NOISE_SWEEP = [0.0, 0.1, 0.2, 0.3]
NOISE_ABLATION_SIGMA = 0.2
MIXUP_ALPHA = 0.4
BATCH_EVAL = 256
BATCH_MONTAGE = 16


class DeepMLP(nn.Module):
    """
    Fully connected neural network used for Task 2 classification on Fashion-MNIST.

    Purpose:
        Defines the multilayer perceptron architecture used to load and evaluate the
        Task 2 checkpoints for the ablation, robustness, and montage experiments.

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
                Scalar in the range [0.0, 1.0].

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


def _try_font(size: int = 14) -> ImageFont.ImageFont:
    """
    Load a font for drawing text on the output montage image.

    Args:
        size (int):
            Requested font size in pixels for titles or tile annotations.

    Returns:
        ImageFont.ImageFont:
            A Pillow font object. The function first tries common system fonts and
            falls back to Pillow's default font if needed.
    """
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def mixup_with_metadata(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """
    Apply MixUp to one batch and return both the mixed inputs and target metadata.

    Args:
        x (torch.Tensor):
            Input batch of images with shape (batch_size, 1, 28, 28).

        y (torch.Tensor):
            Integer class labels with shape (batch_size,).

        alpha (float):
            MixUp Beta-distribution parameter controlling interpolation strength.
            Scalar value greater than 0. If alpha <= 0, the original batch is returned.

    Returns:
        tuple:
            (x_mix, lam, y_a, y_b), where:
            - x_mix is the mixed image batch of shape (batch_size, 1, 28, 28),
            - lam is the sampled interpolation weight as a float,
            - y_a is the original label tensor of shape (batch_size,),
            - y_b is the paired label tensor after batch permutation.
    """
    if alpha <= 0:
        return x, 1.0, y, y
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    x_mix = lam * x + (1.0 - lam) * x2
    return x_mix, lam, y, y2


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device, noise_sigma: float = 0.0) -> float:
    """
    Compute classification accuracy on clean or noise-corrupted test inputs.

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

        noise_sigma (float):
            Standard deviation of Gaussian noise added to the input images before
            evaluation. Scalar value in pixel-normalised space. A value of 0.0 means
            clean evaluation.

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
        if noise_sigma > 0:
            x = x + noise_sigma * torch.randn_like(x)
            x = torch.clamp(x, 0.0, 1.0)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def predict_top1(model: nn.Module, x: torch.Tensor):
    """
    Compute top-1 predictions and confidence scores for an input batch.

    Args:
        model (nn.Module):
            PyTorch classification model used for inference.

        x (torch.Tensor):
            Input batch of images with shape (batch_size, 1, 28, 28).

    Returns:
        tuple:
            (pred, conf), where:
            - pred is a tensor of predicted class indices with shape (batch_size,),
            - conf is a tensor of maximum softmax probabilities with shape (batch_size,).
    """
    model.eval()
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def mean_std(vals: list[float]) -> tuple[float, float]:
    """
    Compute the mean and population standard deviation of a list of scalar values.

    Args:
        vals (list[float]):
            One-dimensional list of numeric values, typically accuracies collected
            across multiple seeds or robustness settings.

    Returns:
        tuple[float, float]:
            A pair (mean, std). If the input list is empty, the function returns
            (0.0, 0.0).
    """
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5


def print_ablation_table(rows, sigma: float) -> None:
    """
    Print the clean/noisy ablation comparison table for Task 2 models.

    Args:
        rows:
            Iterable of row tuples of the form (model_name, clean_acc, noisy_acc),
            where accuracies are floats in the range [0.0, 1.0].

        sigma (float):
            Noise standard deviation used for the noisy evaluation column.

    Returns:
        None.
            The function only prints the formatted table to the terminal.
    """
    print(f"\nA) Ablation (clean vs noisy σ={sigma})")
    print("-" * 34)
    header = f"{'Model':<12} {'Clean Acc':<12} {'Noisy Acc':<12} {'Drop':<10}"
    print(header)
    print("-" * len(header))
    for name, clean, noisy in rows:
        drop = clean - noisy
        print(f"{name:<12} {clean*100:7.2f}%     {noisy*100:7.2f}%     {drop*100:7.2f}%")


def _tensor28_to_pil_gray(img_u8: torch.Tensor) -> Image.Image:
    """
    Convert one 28x28 uint8 image tensor to a Pillow grayscale image.

    Args:
        img_u8 (torch.Tensor):
            Tensor of shape (28, 28) containing uint8 pixel values in the range
            [0, 255].

    Returns:
        Image.Image:
            Pillow grayscale image of size 28x28.
    """
    img_u8 = img_u8.contiguous()
    b = bytes(img_u8.view(-1).tolist())
    return Image.frombytes("L", (28, 28), b)


def save_annotated_montage(out_path: Path, x_mix, lam, y_a, y_b, pred, conf, grid=4) -> None:
    """
    Save an annotated montage showing MixUp inputs, target weights, and predictions.

    Args:
        out_path (Path):
            Output file path for the saved PNG montage.

        x_mix:
            Mixed image tensor of shape (batch_size, 1, 28, 28).

        lam:
            Global MixUp interpolation weight used for the batch.

        y_a:
            Original label tensor of shape (batch_size,).

        y_b:
            Paired label tensor of shape (batch_size,) after permutation.

        pred:
            Predicted class-index tensor of shape (batch_size,).

        conf:
            Prediction-confidence tensor of shape (batch_size,).

        grid:
            Integer montage grid width/height. A value of 4 produces a 4x4 montage.

    Returns:
        None.
            The function saves the montage image to disk and does not return a value.
    """
    n = grid * grid
    x_mix = x_mix[:n].detach().cpu()
    y_a = y_a[:n].detach().cpu()
    y_b = y_b[:n].detach().cpu()
    pred = pred[:n].detach().cpu()
    conf = conf[:n].detach().cpu()

    scale = 4
    img_w = 28 * scale
    img_h = 28 * scale
    text_h = 46
    tile_w = img_w
    tile_h = img_h + text_h

    pad = 10
    title_h = 60

    W = grid * tile_w + (grid + 1) * pad
    H = title_h + grid * tile_h + (grid + 1) * pad

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _try_font(18)
    text_font = _try_font(12)

    draw.text((pad, 10), f"MixUp montage (alpha={MIXUP_ALPHA}, lambda≈{lam:.2f})", fill="black", font=title_font)
    draw.text((pad, 32), "Tile text: target weights + model prediction", fill="black", font=text_font)

    for i in range(n):
        r = i // grid
        c = i % grid
        x0 = pad + c * (tile_w + pad)
        y0 = title_h + pad + r * (tile_h + pad)

        img_u8 = (x_mix[i].squeeze(0) * 255.0).clamp(0, 255).byte()
        pil = _tensor28_to_pil_gray(img_u8).resize((img_w, img_h), resample=Image.NEAREST).convert("RGB")
        canvas.paste(pil, (x0, y0))

        ya = int(y_a[i].item()); yb = int(y_b[i].item())
        tgt = f"tgt: {FASHION_CLASSES[ya][:10]} {lam:.2f} | {FASHION_CLASSES[yb][:10]} {1-lam:.2f}"

        p = int(pred[i].item()); c1 = float(conf[i].item())
        pr = f"pred: {FASHION_CLASSES[p][:12]} ({c1:.2f})"

        ty = y0 + img_h + 2
        draw.text((x0, ty), tgt, fill="black", font=text_font)
        draw.text((x0, ty + 18), pr, fill="black", font=text_font)

    canvas.save(out_path)
    print(f"Saved: {out_path.name}")


def main() -> None:
    """
    Execute the full Task 2 evaluation pipeline.

    Purpose:
        Loads the Fashion-MNIST test set, evaluates clean and noisy performance for
        the Task 2 ablations, computes Multi-seed reliability analysis (not performed in this
        run due to unavailable additional seed checkpoints) for MixUp+Label
        Smoothing, computes the robustness sweep across multiple noise levels, saves
        an annotated MixUp montage, and prints the supporting technical justification.

    Inputs:
        None directly from function arguments.
        The function reads:
        - ablation_history.json
        - saved checkpoint files such as none_best.pth, mixup_best.pth, ls_best.pth,
        and mixup_ls_best.pth

    Returns:
        None.
            The function prints the evaluation summary and saves robustness_demo.png.
    """
    here = Path(__file__).resolve().parent
    device = torch.device("cpu")

    # Load test data
    data_dir = here / "data"
    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)

    # Load ablation history to discover available checkpoints/seeds
    hist_path = here / "ablation_history.json"
    if not hist_path.exists():
        raise FileNotFoundError("ablation_history.json not found. Run task2/train.py first.")

    with open(hist_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    # Collect checkpoints available
    # seed0 expected: none_best.pth, mixup_best.pth, ls_best.pth, mixup_ls_best.pth
    ckpts = {
        "none": here / "none_best.pth",
        "mixup": here / "mixup_best.pth",
        "ls": here / "ls_best.pth",
        "mixup+ls": here / "mixup_ls_best.pth",
    }

    # Also multi-seed main checkpoints
    seed_ckpts = []
    for seed in blob["meta"].get("seeds", [0, 1, 2]):
        p = here / ("mixup_ls_best.pth" if seed == 0 else f"mixup_ls_seed{seed}_best.pth")
        if p.exists():
            seed_ckpts.append((seed, p))

    # ------------------------
    # A) Ablation table (seed0)
    # ------------------------
    ablation_rows = []
    for name, p in ckpts.items():
        if not p.exists():
            continue
        m = DeepMLP(0.10).to(device)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        clean = accuracy(m, test_loader, device, noise_sigma=0.0)
        noisy = accuracy(m, test_loader, device, noise_sigma=NOISE_ABLATION_SIGMA)
        ablation_rows.append((name, clean, noisy))
    print_ablation_table(ablation_rows, NOISE_ABLATION_SIGMA)

    # ----------------------------------------
    # B) Multi-seed reliability for MixUp+LS
    # ----------------------------------------
    if len(seed_ckpts) >= 2:
        clean_list = []
        noisy_list = []
        for seed, p in seed_ckpts:
            m = DeepMLP(0.10).to(device)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            clean_list.append(accuracy(m, test_loader, device, 0.0))
            noisy_list.append(accuracy(m, test_loader, device, NOISE_ABLATION_SIGMA))

        mc, sc = mean_std(clean_list)
        mn, sn = mean_std(noisy_list)
        print("\nB) Multi-seed reliability (MixUp+LS, if checkpoints available) — mean±std")
        print("------------------------------------------------")
        print(f"Seeds available: {[s for s,_ in seed_ckpts]}")
        print(f"Clean acc: {mc*100:.2f}±{sc*100:.2f}%")
        print(f"Noisy acc (σ={NOISE_ABLATION_SIGMA}): {mn*100:.2f}±{sn*100:.2f}%")
        print(f"Drop: {(mc-mn)*100:.2f}% (computed on means)")
    else:
        print("\nB) Multi-seed reliability was not evaluated due to unavailable additional seed checkpoints).")

    # ----------------------------------------
    # C) Robustness sweep (mean±std if seeds)
    # ----------------------------------------
    print("\nC) Robustness sweep for MixUp+LS (accuracy vs σ)")
    print("-------------------------------------------------")
    print(f"{'sigma':<8} {'acc':<18} {'drop from clean':<18}")
    print("-" * 46)

    # choose models for sweep: per-seed if available, otherwise just seed0
    sweep_models = []
    if seed_ckpts:
        for seed, p in seed_ckpts:
            m = DeepMLP(0.10).to(device)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            sweep_models.append((seed, m))
    else:
        # fallback
        p = here / "mixup_ls_best.pth"
        if p.exists():
            m = DeepMLP(0.10).to(device)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            sweep_models.append((0, m))

    # compute clean mean (for drop baseline)
    clean_vals = [accuracy(m, test_loader, device, 0.0) for _, m in sweep_models]
    clean_m, clean_s = mean_std(clean_vals)

    for s in NOISE_SWEEP:
        vals = [accuracy(m, test_loader, device, s) for _, m in sweep_models]
        m_acc, s_acc = mean_std(vals)
        drop_m = clean_m - m_acc
        print(f"{s:<8.2f} {m_acc*100:6.2f}±{s_acc*100:5.2f}%      {drop_m*100:6.2f}%")

    # ----------------------------------------
    # Montage (use seed0 model if exists)
    # ----------------------------------------
    main_model_path = here / "mixup_ls_best.pth"
    if not main_model_path.exists():
        # fallback to any
        main_model_path = seed_ckpts[0][1] if seed_ckpts else None

    if main_model_path is not None:
        main_model = DeepMLP(0.10).to(device)
        main_model.load_state_dict(torch.load(main_model_path, map_location="cpu"))

        demo_loader = DataLoader(test_ds, batch_size=BATCH_MONTAGE, shuffle=True, num_workers=0)
        x, y = next(iter(demo_loader))
        x = x.to(device); y = y.to(device)

        x_mix, lam, y_a, y_b = mixup_with_metadata(x, y, MIXUP_ALPHA)
        pred, conf = predict_top1(main_model, x_mix)
        save_annotated_montage(here / "robustness_demo.png", x_mix, lam, y_a, y_b, pred, conf, grid=4)

    # ----------------------------------------
    # Concise theory (less generic, tied to outputs)
    # ----------------------------------------
    print("\n" + "=" * 92)
    print("TASK 2 — Technical Justification (concise, evidence-backed)")
    print("=" * 92)
    print(f"""

MixUp training works by replacing individual training samples with convex combinations
of two samples drawn from the same batch. Concretely, given two images x_i and x_j and
their one-hot label vectors y_i and y_j, the model is trained on the blended input
x_mix = lambda * x_i + (1 - lambda) * x_j with the correspondingly blended target
y_mix = lambda * y_i + (1 - lambda) * y_j, where lambda is sampled from a Beta
distribution with both parameters equal to alpha. The consequence for generalisation
is that the model is less able to rely on memorisation of individual training points,
because the loss is defined over a continuous family of interpolations rather than a
fixed set of discrete examples. Any decision boundary that perfectly separates the
original training points will still incur a cost if it fails to interpolate smoothly
between them. This is the mechanism by which MixUp discourages memorization: it makes
exact memorisation of individual examples insufficient for minimising the training loss.

The annotated robustness_demo.png saved by this script provides direct visual evidence
of this behaviour. Each tile shows a mixed image alongside the two source class labels
weighted by lambda and 1 - lambda, together with the model's prediction and confidence
on that mixed input. The model is therefore evaluated on inputs that were never seen
during training as pure examples, which is precisely the property MixUp is designed to
encourage: learning decision boundaries that behave sensibly between training samples.

Label smoothing addresses a related but distinct issue. When training with hard one-hot
targets, the cross-entropy loss encourages the predicted probability for the correct
class to approach 1.0 and all others to approach 0.0. This drives the logits toward
extreme values and leads to overconfident predictions. Overconfident models tend to fit
the training distribution too sharply rather than learning representations that transfer
well to unseen data. Label smoothing replaces the hard targets with softened ones: the
correct class is assigned a target of 1 - epsilon, while the remaining probability mass
epsilon is distributed uniformly across all K classes. This reduces the pressure toward
extreme logits and encourages more moderate, better-calibrated predictions.

The robustness sweep across noise levels sigma in {NOISE_SWEEP} provides a more complete
picture of generalisation than a single clean-accuracy value. A model that has effectively
memorised the training set will typically show a steep drop in accuracy as soon as noise
is introduced, because its decision boundaries are tightly fitted to the original pixel
values. In contrast, a model trained with MixUp and label smoothing shows in the reported 
results a more gradual degradation, reflecting smoother behaviour in the input space.

In the results here, the most informative signal is the change in accuracy as noise
increases. The ablation table at sigma equal to {NOISE_ABLATION_SIGMA} isolates the
contribution of each component by comparing none, MixUp only, label smoothing only, and
the combined model on the same noisy test set. The drop relative to clean accuracy is
particularly important: a smaller drop indicates that the learned representation is more
stable under input perturbations. The combined MixUp and label smoothing model shows the 
smallest drop in the reported results under this criterion, since both techniques explicitly 
encourage smoother and less overconfident decision boundaries.

In the observed results, the combined MixUp and label smoothing configuration shows
the strongest robustness behaviour, with a noticeably smaller drop in accuracy as
noise increases compared to the baseline. The baseline model degrades rapidly even
under moderate noise, while MixUp provides a smaller drop in the reported results; label
smoothing alone does not improve robustness in this setting. The combined model is
the most stable among the evaluated configurations at σ = 0.2, and it also shows a
gradual degradation across its own robustness sweep. This supports the interpretation
that both interpolation-based training and softened targets contribute to smoother
decision boundaries and improved robustness.
""")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()