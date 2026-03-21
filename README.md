# Applied Deep Learning
## Generalization Dynamics & Robust Representation Learning (PyTorch)

This repository showcases practical deep learning experimentation focused on generalization behaviour, regularisation techniques, and robust model representations.

The project demonstrates the ability to:
* design neural networks from scratch
* implement research-level regularisation techniques
* analyse bias–variance trade-offs empirically
* produce reproducible ML experiments
* communicate technical findings clearly

Built using PyTorch with minimal dependencies to ensure transparency and reproducibility.

---
### Highlights
* Deep neural network constructed manually (no pretrained architectures)
* Controlled experiments analysing the generalization gap
* Custom implementation of MixUp augmentation
* Custom implementation of Label Smoothing loss
* Early stopping implemented from first principles
* Robustness evaluation under noisy inputs
* Reproducible training and evaluation pipeline
* Clean, structured code with consistent docstrings

---
### Project Structure

    cw1-pt/
    │
    ├── task1/
    │   ├── train.py
    │   ├── task.py
    │   ├── baseline_model.pth
    │   ├── regularized_model.pth
    │   └── generalization_gap.png
    │
    ├── task2/
    │   ├── train.py
    │   ├── task.py
    │   ├── mixup_ls_best.pth
    │   └── robustness_demo.png

**Each task includes:**
* training pipeline
* evaluation script
* saved model weights
* generated visual outputs
* technical analysis printed to terminal

---
### Task 1 — Generalization Behaviour
Investigates how model capacity and regularisation influence performance on unseen data.

**Key ideas explored:**
* bias–variance trade-off
* implicit vs explicit regularisation
* optimization path effects
* model capacity control

**Output:**
* training vs validation performance comparison
* generalization gap visualisation
* technical interpretation of experimental behaviour

---
### Task 2 — Robust Representations
Implements robustness-oriented regularisation techniques directly from tensor operations.

**Techniques implemented:**
* **MixUp**: Interpolates training samples to encourage smoother decision boundaries.
* **Label Smoothing**: Reduces overconfidence by softening target distributions.

**Additional evaluation:**
* noise perturbation testing
* qualitative robustness visualisation

---
### Environment
Designed to run in a controlled micromamba environment:

    micromamba create --name comp0197-pt python=3.12 -y
    micromamba activate comp0197-pt
    pip install torch torchvision pillow --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

---
### Running the Code

**Train models:**

    cd cw1-pt/task1
    python train.py
    cd ../task2
    python train.py

**Run evaluation:**

    cd cw1-pt/task1
    python task.py
    cd ../task2
    python task.py

> **Note:** Outputs are saved automatically as PNG files inside each task directory.

---
### Technical Skills Demonstrated
* PyTorch model development
* neural network architecture design
* experimental ML methodology
* performance analysis
* regularisation strategies
* reproducible research workflows
* clean software structure
* technical documentation
