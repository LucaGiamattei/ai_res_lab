"""Build challenge_2_adversarial/{starter,solution}.ipynb."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import bootstrap_cell, code, make_nb, md, write

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "challenge_2_adversarial"


# ---------- Cells -----------------------------------------------------------------------------------

INTRO_MD = r"""
# Challenge 2 — Rompi il modello

**Tema:** generare *esempi avversariali* (FGSM, Goodfellow et al. 2015) contro un classificatore ImageNet pre-addestrato (`MobileNetV2`) e operazionalizzare la nozione di "robustezza" tramite un budget di perturbazione esplicito.

**Pertinenza normativa.** L'**Art. 15** del Regolamento (UE) 2024/1689 impone, per i sistemi ad alto rischio, un livello "appropriato" di **accuratezza, robustezza e cibersicurezza**. In particolare il §5 specifica che i sistemi devono essere "*resilienti rispetto a tentativi di terzi non autorizzati di alterarne l'uso, gli output o le prestazioni sfruttando vulnerabilità del sistema*". Le perturbazioni avversariali sono il caso paradigmatico di questa minaccia.

**Obiettivo.** In ~30 minuti:

1. Carichiamo MobileNetV2 e verifichiamo predizioni baseline su 5 immagini ImageNet.
2. Implementiamo FGSM nella sua forma canonica:
   $$\hat{x}_{\mathrm{adv}} = \mathrm{clip}\bigl(x + \varepsilon \cdot \mathrm{sign}\bigl(\nabla_x \mathcal{L}(\theta, x, y)\bigr), 0, 1\bigr)$$
   dove $\mathcal{L}$ è la cross-entropy e $\varepsilon$ il *budget* di perturbazione $L_\infty$.
3. Eseguiamo uno sweep su $\varepsilon \in \{0.001, 0.005, 0.01, 0.02, 0.05, 0.1\}$.
4. Definiamo la *robustezza* come $\varepsilon_{\max}$ per cui l'accuratezza resta $\geq 0.80$, e compiliamo l'evidence row.

**FGSM è l'attacco più *debole* della letteratura.** Una valutazione di robustezza rigorosa richiederebbe PGD (Madry et al. 2018) e AutoAttack (Croce & Hein 2020). Qui FGSM serve da *baseline pedagogico*: se il modello cade contro FGSM, cade a fortiori contro attacchi più sofisticati.

**Riferimenti:**
- Goodfellow, Shlens, Szegedy. *Explaining and Harnessing Adversarial Examples.* ICLR 2015.
- Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
- Croce, Hein. *Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.* ICML 2020.
"""

CELL_INSTALL = """\
%pip install -q "torch>=2.2" "torchvision>=0.17" "pillow>=10" matplotlib
"""

CELL_IMPORTS = """\
import os
import sys
import random
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"  # MobileNetV2 is fast on CPU; Colab Free Tier-friendly.

# PACKAGE_ROOT is set by the bootstrap cell above.
DATA_DIR     = PACKAGE_ROOT / "challenge_2_adversarial" / "data"
IMG_DIR      = DATA_DIR / "images"
CLASSES_FILE = DATA_DIR / "imagenet_classes.txt"
EVIDENCE_CSV = PACKAGE_ROOT / "shared" / "evidence_template.csv"

print(f"Device: {DEVICE}")
print(f"Images: {IMG_DIR}")
"""

CELL_LOAD_MODEL = """\
# Load MobileNetV2 pretrained on ImageNet (~14 MB download on first run).
weights = MobileNet_V2_Weights.IMAGENET1K_V2
base_model = mobilenet_v2(weights=weights).to(DEVICE).eval()

# Load the class labels.
classes = [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]
assert len(classes) == 1000, f"Expected 1000 classes, got {len(classes)}"
print(f"Loaded MobileNetV2 ({sum(p.numel() for p in base_model.parameters())/1e6:.2f} M params)")
print(f"Classes loaded: {len(classes)}")
"""

CELL_NORMALIZED_MODEL = """\
# We work with images in [0, 1] pixel space (easier to visualize and to bound
# the L-inf perturbation). The model expects ImageNet-normalized input, so we
# wrap it: NormalizedModel(x) = base_model((x - mean) / std).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

class NormalizedModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        return self.base((x - IMAGENET_MEAN) / IMAGENET_STD)

model = NormalizedModel(base_model).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False  # we only need gradients w.r.t. the input

print("Model wrapped: input now expected in [0, 1] pixel space.")
"""

CELL_PREPROCESS_HELPER = """\
# Preprocessing pipeline that produces a [0, 1] tensor of shape (1, 3, 224, 224).
preprocess_to_01 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # [0, 1]
])

def load_image(name: str) -> torch.Tensor:
    img = Image.open(IMG_DIR / name).convert("RGB")
    return preprocess_to_01(img).unsqueeze(0).to(DEVICE)

def predict(x: torch.Tensor, k: int = 5):
    \"\"\"Return list of (class_id, class_name, prob) sorted by descending prob.\"\"\"
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    top = torch.topk(probs, k)
    return [(int(i), classes[int(i)], float(p)) for i, p in zip(top.indices, top.values)]

# Tensor representation utilities for matplotlib display.
def to_displayable(x: torch.Tensor) -> np.ndarray:
    return x.detach().clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
"""

CELL_BASELINE_PREDICT = """\
# Baseline predictions on 5 sample images.
SAMPLES = [
    ("panda.jpg",            388, "giant panda"),
    ("school_bus.jpg",       779, "school bus"),
    ("golden_retriever.jpg", 207, "golden retriever"),
    ("traffic_light.jpg",    920, "traffic light"),
    ("espresso.jpg",         967, "espresso"),
]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, (fname, expected_id, expected_name) in zip(axes, SAMPLES):
    x = load_image(fname)
    top1 = predict(x, k=1)[0]
    ax.imshow(to_displayable(x))
    ax.axis("off")
    ok = "OK" if top1[0] == expected_id else "MISS"
    ax.set_title(f"{fname}\\n{top1[1]}\\np={top1[2]:.2f} [{ok}]", fontsize=8)
plt.tight_layout()
plt.show()
"""

# TODO 1 — fgsm_attack
TODO1_MD = r"""
## TODO 1 — Implementate `fgsm_attack`

Date un'immagine $x \in [0,1]^{1 \times 3 \times H \times W}$ e la sua label vera $y$, FGSM produce:

$$\hat{x}_{\mathrm{adv}} = \mathrm{clip}\bigl(x + \varepsilon \cdot \mathrm{sign}\bigl(\nabla_x \mathcal{L}(\theta, x, y)\bigr), 0, 1\bigr)$$

Hint operativi (in commento qui sotto):
- `image.requires_grad_(True)` per registrare il gradiente
- `loss = F.cross_entropy(logits, label)`
- `loss.backward()` calcola il gradiente
- il gradiente è in `image.grad`
- usare `.sign()` per ottenere il sign-tensor

Restituire un tensore della stessa forma e dtype dell'input, **scollegato** dal grafo computazionale (`detach()`).

Vincolo importante: applicare `clamp(0, 1)` per restare in pixel-space valido. Senza clamp, eps grandi producono pixel out-of-range che non sono immagini reali.
"""

TODO1_STARTER = """\
def fgsm_attack(model: nn.Module, image: torch.Tensor, label: torch.Tensor, epsilon: float) -> torch.Tensor:
    \"\"\"FGSM attack in [0, 1] pixel space.

    Args:
        model: differentiable classifier; expects input shape (B, 3, H, W) in [0, 1].
        image: input tensor (1, 3, H, W) in [0, 1].
        label: ground-truth class id, shape (1,).
        epsilon: L-infinity perturbation budget.

    Returns:
        adv: adversarial image, same shape and dtype as `image`, in [0, 1].
    \"\"\"
    # TODO: implement FGSM. Hints:
    # 1. image = image.clone().detach().requires_grad_(True)
    # 2. logits = model(image); loss = F.cross_entropy(logits, label)
    # 3. model.zero_grad(); loss.backward()
    # 4. perturb = epsilon * image.grad.sign()
    # 5. adv = (image + perturb).clamp(0, 1).detach()
    raise NotImplementedError("Implementare fgsm_attack")
"""

TODO1_SOLUTION = """\
def fgsm_attack(model: nn.Module, image: torch.Tensor, label: torch.Tensor, epsilon: float) -> torch.Tensor:
    \"\"\"FGSM attack in [0, 1] pixel space.\"\"\"
    image = image.clone().detach().requires_grad_(True)
    logits = model(image)
    loss = F.cross_entropy(logits, label)
    model.zero_grad()
    loss.backward()
    perturb = epsilon * image.grad.sign()
    adv = (image + perturb).clamp(0.0, 1.0).detach()
    return adv
"""

CELL_VERIFY_TODO1 = """\
# Verify: shape, dtype, range.
x = load_image("panda.jpg")
y = torch.tensor([388], device=DEVICE)  # giant panda

x_adv = fgsm_attack(model, x, y, epsilon=0.01)
assert x_adv.shape == x.shape,           f"Shape mismatch: {x_adv.shape} vs {x.shape}"
assert x_adv.dtype == x.dtype,           f"Dtype mismatch: {x_adv.dtype} vs {x.dtype}"
assert x_adv.min() >= 0 - 1e-6,          f"Pixel < 0: {x_adv.min()}"
assert x_adv.max() <= 1 + 1e-6,          f"Pixel > 1: {x_adv.max()}"
linf = (x_adv - x).abs().max().item()
assert linf <= 0.01 + 1e-6,              f"L-inf budget violato: {linf}"
print(f"OK: shape={tuple(x_adv.shape)}, L-inf perturbation = {linf:.6f}")
"""

# TODO 2 — single attack
TODO2_MD = """\
## TODO 2 — Attacco singolo sul panda con `epsilon=0.01`

1. Caricate `panda.jpg`.
2. Eseguite `fgsm_attack` con `epsilon=0.01`.
3. Visualizzate (asse 1×3): originale | perturbazione amplificata 50× | adversarial.
4. Stampate la top-1 prediction su entrambi (originale e adversarial).
"""

TODO2_STARTER = """\
# TODO 2: single FGSM attack on panda.jpg with epsilon=0.01.

EPSILON = 0.01
x        = load_image("panda.jpg")
y        = torch.tensor([388], device=DEVICE)

x_adv    = None  # TODO: chiamare fgsm_attack(model, x, y, EPSILON)

if x_adv is None:
    raise NotImplementedError("Eseguire l'attacco e salvare il risultato in x_adv.")

perturb_amplified = (x_adv - x) * 50.0 + 0.5  # for visualization

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(to_displayable(x));                axes[0].set_title("Originale")
axes[1].imshow(to_displayable(perturb_amplified)); axes[1].set_title("Perturbazione (×50, +0.5)")
axes[2].imshow(to_displayable(x_adv));            axes[2].set_title(f"Adversarial (eps={EPSILON})")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()

p_orig = predict(x, k=1)[0]
p_adv  = predict(x_adv, k=1)[0]
print(f"Originale: id={p_orig[0]:>4d} '{p_orig[1]}' p={p_orig[2]:.3f}")
print(f"Adversarial: id={p_adv[0]:>4d} '{p_adv[1]}' p={p_adv[2]:.3f}")
print(f"Predizione cambiata: {p_orig[0] != p_adv[0]}")
"""

TODO2_SOLUTION = """\
EPSILON = 0.01
x        = load_image("panda.jpg")
y        = torch.tensor([388], device=DEVICE)

x_adv    = fgsm_attack(model, x, y, EPSILON)

perturb_amplified = (x_adv - x) * 50.0 + 0.5

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(to_displayable(x));                axes[0].set_title("Originale")
axes[1].imshow(to_displayable(perturb_amplified)); axes[1].set_title("Perturbazione (\\u00d750, +0.5)")
axes[2].imshow(to_displayable(x_adv));            axes[2].set_title(f"Adversarial (eps={EPSILON})")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()

p_orig = predict(x, k=1)[0]
p_adv  = predict(x_adv, k=1)[0]
print(f"Originale: id={p_orig[0]:>4d} '{p_orig[1]}' p={p_orig[2]:.3f}")
print(f"Adversarial: id={p_adv[0]:>4d} '{p_adv[1]}' p={p_adv[2]:.3f}")
print(f"Predizione cambiata: {p_orig[0] != p_adv[0]}")
"""

# TODO 3 — sweep
TODO3_MD = """\
## TODO 3 — Sweep su $\\varepsilon$

Per ciascun valore di $\\varepsilon \\in \\{0.001, 0.005, 0.01, 0.02, 0.05, 0.1\\}$ e per ciascuna delle 5 immagini:

1. Generate l'esempio avversariale con `fgsm_attack`.
2. Verificate se la top-1 prediction è cambiata rispetto alla label vera.
3. Misurate la perturbazione $L_\\infty$ effettiva.

Tabulate i risultati in un dizionario `results[eps] = {"acc": ..., "linf": ...}` dove `acc` è la frazione di immagini ancora correttamente classificate e `linf` è la media delle perturbazioni.

Suggerimento: usate `EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]` e iterate.
"""

TODO3_STARTER = """\
# TODO 3: epsilon sweep over the 5 sample images.

EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
results = {}  # eps -> {"acc": float, "linf": float, "flips": list[bool]}

# TODO: loop over EPSILONS and SAMPLES; for each combo, run FGSM, check correctness, record metrics.

if not results:
    raise NotImplementedError("Compilare il dizionario results.")

print(f"{'eps':<10}{'acc':<10}{'mean L-inf':<15}")
for eps in EPSILONS:
    print(f"{eps:<10}{results[eps]['acc']:<10.3f}{results[eps]['linf']:<15.5f}")
"""

TODO3_SOLUTION = """\
EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
results = {}

for eps in EPSILONS:
    flips = []
    linfs = []
    for fname, expected_id, _ in SAMPLES:
        x = load_image(fname)
        y = torch.tensor([expected_id], device=DEVICE)
        x_adv = fgsm_attack(model, x, y, eps)
        adv_pred_id = predict(x_adv, k=1)[0][0]
        flipped = adv_pred_id != expected_id
        flips.append(flipped)
        linfs.append((x_adv - x).abs().max().item())
    results[eps] = {
        "acc": 1.0 - sum(flips) / len(flips),
        "linf": float(np.mean(linfs)),
        "flips": flips,
    }

print(f"{'eps':<10}{'acc':<10}{'mean L-inf':<15}")
for eps in EPSILONS:
    print(f"{eps:<10}{results[eps]['acc']:<10.3f}{results[eps]['linf']:<15.5f}")
"""

CELL_VIZ_GRID = """\
# Visualization scaffold: 6x3 grid of (original / perturbation / adversarial) on the panda
# at each epsilon, plus an accuracy-vs-epsilon line chart.

panda_x = load_image("panda.jpg")
panda_y = torch.tensor([388], device=DEVICE)

fig, axes = plt.subplots(len(EPSILONS), 3, figsize=(9, 2.6 * len(EPSILONS)))
for row, eps in enumerate(EPSILONS):
    x_adv = fgsm_attack(model, panda_x, panda_y, eps)
    perturb_amp = (x_adv - panda_x) * 50.0 + 0.5
    p_adv = predict(x_adv, k=1)[0]
    axes[row, 0].imshow(to_displayable(panda_x));   axes[row, 0].set_title(f"eps={eps}\\nOriginale", fontsize=9)
    axes[row, 1].imshow(to_displayable(perturb_amp));axes[row, 1].set_title("Perturbazione (\\u00d750)", fontsize=9)
    axes[row, 2].imshow(to_displayable(x_adv));      axes[row, 2].set_title(f"Adv -> {p_adv[1]} ({p_adv[2]:.2f})", fontsize=9)
    for c in range(3):
        axes[row, c].axis("off")
plt.tight_layout()
plt.savefig(PACKAGE_ROOT / "challenge_2_adversarial" / "img" / "panda_eps_sweep.png", dpi=110, bbox_inches="tight")
plt.show()

# Line chart: accuracy vs epsilon.
fig, ax = plt.subplots(figsize=(7, 4))
xs = list(results.keys())
ys = [results[e]["acc"] for e in xs]
ax.plot(xs, ys, marker="o", color="#CC3311")
ax.set_xscale("log")
ax.set_xlabel("epsilon (L-inf budget)")
ax.set_ylabel("accuracy on 5 samples")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.8, ls="--", color="grey", label="target 80%")
ax.set_title("MobileNetV2 — FGSM accuracy vs perturbation budget")
ax.legend()
plt.tight_layout()
plt.savefig(PACKAGE_ROOT / "challenge_2_adversarial" / "img" / "eps_sweep.png", dpi=110, bbox_inches="tight")
plt.show()
"""

# TODO 4 — evidence row
TODO4_MD = """\
## TODO 4 — Definite la *robustezza operativa* e compilate l'evidence row

Date i vostri `results`, definite la *robustezza operativa* come:

$$\\varepsilon_{\\max}^{(80\\%)} = \\max\\Bigl\\{\\varepsilon : \\mathrm{acc}(\\varepsilon') \\geq 0.80 \\,\\,\\forall\\, \\varepsilon' \\leq \\varepsilon\\Bigr\\}$$

cioè il più grande budget di perturbazione per cui l'accuratezza resta $\\geq 0.80$ **in maniera monotona** (dal valore più piccolo testato fino a $\\varepsilon$ incluso). Se l'accuratezza scende sotto $0.80$ già al primo $\\varepsilon$ testato, $\\varepsilon_{\\max} = 0$ — il modello non è robusto al budget testato più piccolo.

> Perché la versione monotona? FGSM è un attacco *single-step*: piccoli eps e grandi eps possono incidentalmente non flippare gli stessi sample (effetto noto di non-monotonia di FGSM, vedi Madry et al. 2018). Il valore *monotonia-corretto* è una stima conservativa della robustezza ed è l'unica interpretazione coerente con la pretesa "il modello resiste fino a $\\varepsilon$".

Compilate l'evidence row qui sotto. La soglia operativa che proponiamo è $\\varepsilon_{\\max} \\geq 0.05$: se non la raggiunge, `status=fail`.

> 0.05 è una soglia *arbitraria scelta a fini didattici*. In un audit reale la soglia va scelta per criterio operativo (es. budget di perturbazione che un attaccante può realisticamente iniettare via JPEG compression, o tramite stickers fisici). Non esistono soglie regolatorie consolidate per FGSM in ImageNet — è materia di ricerca attiva.
"""

TODO4_STARTER = """\
# TODO 4: compute eps_max @ 80% accuracy and write the evidence row.

THRESHOLD_ACC = 0.80
THRESHOLD_EPS = 0.05  # operational threshold (educational choice)

eps_max = None  # TODO: monotone eps_max — the largest eps such that for ALL e <= eps,
                #       results[e]["acc"] >= THRESHOLD_ACC.
                #       Iterate sorted(EPSILONS) and break as soon as one fails.
                #       If even the smallest eps fails, set eps_max = 0.0.

if eps_max is None:
    raise NotImplementedError("Calcolare eps_max")

status = "pass" if eps_max >= THRESHOLD_EPS else "fail"
NOTES = ""  # TODO: una frase, specifica.

if not NOTES:
    raise NotImplementedError("Compilare NOTES con un'osservazione specifica (1 frase).")

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
import csv
row = dict(
    challenge="C2",
    system="ImageNet-MobileNetV2",
    metric="Adversarial robustness (FGSM, L-inf, eps_max @ 80% acc)",
    threshold=f">= {THRESHOLD_EPS}",
    observed=f"{eps_max:.4f}",
    status=status,
    mitigation="none",
    notes=NOTES,
    timestamp=ts,
)
EVIDENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
write_header = not EVIDENCE_CSV.exists() or EVIDENCE_CSV.stat().st_size == 0
with EVIDENCE_CSV.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"eps_max @ 80% acc = {eps_max:.4f}  ->  status={status}")
print(f"Wrote evidence row to {EVIDENCE_CSV}")
"""

TODO4_SOLUTION = """\
THRESHOLD_ACC = 0.80
THRESHOLD_EPS = 0.05

# Compute eps_max with the monotone interpretation.
eps_max = 0.0
for eps in sorted(EPSILONS):
    if results[eps]["acc"] >= THRESHOLD_ACC:
        eps_max = eps
    else:
        break

status = "pass" if eps_max >= THRESHOLD_EPS else "fail"
flipped_classes = []
for eps in sorted(results, reverse=True):
    if results[eps]["acc"] < 1.0:
        flipped = [SAMPLES[i][2] for i, f in enumerate(results[eps]["flips"]) if f]
        flipped_classes = flipped
        break
NOTES = (
    f"Robustezza FGSM bassa: gia' a eps={min(EPSILONS):.3f} l'accuratezza scende a {results[min(EPSILONS)]['acc']:.2f}. "
    f"Classi flipped per prime: {', '.join(flipped_classes[:3]) if flipped_classes else 'nessuna'}. "
    f"FGSM e' attacco debole; PGD/AutoAttack peggiorerebbero ulteriormente."
)

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
import csv
row = dict(
    challenge="C2",
    system="ImageNet-MobileNetV2",
    metric="Adversarial robustness (FGSM, L-inf, eps_max @ 80% acc)",
    threshold=f">= {THRESHOLD_EPS}",
    observed=f"{eps_max:.4f}",
    status=status,
    mitigation="none",
    notes=NOTES,
    timestamp=ts,
)
EVIDENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
write_header = not EVIDENCE_CSV.exists() or EVIDENCE_CSV.stat().st_size == 0
with EVIDENCE_CSV.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"eps_max @ 80% acc = {eps_max:.4f}  ->  status={status}")
print(f"Wrote evidence row to {EVIDENCE_CSV}")
"""

CLOSING_MD = """\
## Chiusura — Cosa avete *non* fatto

FGSM è un attacco *single-step*. Esistono attacchi più potenti che dovreste conoscere ma che esulano dal laboratorio:

1. **PGD (Projected Gradient Descent)** — Madry et al. 2018. Iterativo: applica FGSM più volte con piccoli step e proietta nella ball $L_\\infty$. È l'attacco di riferimento per allenamento adversariale.
2. **AutoAttack** — Croce & Hein 2020. Ensemble di 4 attacchi (APGD-CE, APGD-DLR, FAB, Square) parameter-free. Standard de-facto per la valutazione di robustezza.
3. **Attacchi mirati (targeted)** — anziché massimizzare la loss della classe vera, minimizzano la loss verso una classe scelta. Concettualmente FGSM su $-\\nabla_x \\mathcal{L}(\\theta, x, y_\\text{target})$.

**Implicazioni di Art. 15.** Il vostro evidence row dichiara robustezza vs FGSM; un fascicolo Annex IV professionale userebbe almeno PGD-20 e AutoAttack come metodi di valutazione, e includerebbe più di 5 immagini (tipicamente 1000+ dal validation set).

> Per chi volesse approfondire: [robustbench.github.io](https://robustbench.github.io/) mantiene un leaderboard di modelli e attacchi standardizzati.
"""

STRETCH_MD = """\
## Stretch goals (solo soluzione)

### S1 — FGSM mirato (targeted)

Forziamo il modello a predire una classe specifica (`target_id`), non solo "qualcos'altro che non sia la classe vera". Implementazione: sign del gradiente *negativo* della loss verso `target_id`.

### S2 — PGD basico (iterativo, untargeted)

K passi di FGSM, ciascuno con step $\\alpha < \\varepsilon$, e proiezione finale sulla ball $L_\\infty(\\varepsilon)$ centrata in $x$.
"""

STRETCH_S1 = """\
# Stretch S1: targeted FGSM on the panda. Force the model to predict 'soccer ball' (id 805).

def targeted_fgsm(model, image, target_label, epsilon):
    image = image.clone().detach().requires_grad_(True)
    logits = model(image)
    loss = F.cross_entropy(logits, target_label)
    model.zero_grad()
    loss.backward()
    # NOTE the minus sign: we descend the loss towards target.
    perturb = -epsilon * image.grad.sign()
    return (image + perturb).clamp(0.0, 1.0).detach()

panda_x = load_image("panda.jpg")
target = torch.tensor([805], device=DEVICE)  # soccer ball

x_adv = targeted_fgsm(model, panda_x, target, epsilon=0.03)
top5 = predict(x_adv, k=5)
print("Top-5 after targeted FGSM (target='soccer ball'):")
for cid, name, p in top5:
    print(f"  {cid:4d}  {name:30s}  {p:.3f}")
"""

STRETCH_S2 = """\
# Stretch S2: basic PGD untargeted on the panda.

def pgd_attack(model, image, label, epsilon, alpha, steps):
    x_adv = image.clone().detach() + torch.empty_like(image).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, label)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        # Project back to L-inf ball around the original image.
        x_adv = torch.max(torch.min(x_adv, image + epsilon), image - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()

panda_x = load_image("panda.jpg")
panda_y = torch.tensor([388], device=DEVICE)

for steps in [1, 5, 20]:
    adv = pgd_attack(model, panda_x, panda_y, epsilon=0.01, alpha=0.002, steps=steps)
    top1 = predict(adv, k=1)[0]
    print(f"PGD-{steps:>2d} (eps=0.01): {top1[1]} (p={top1[2]:.3f})")
"""


# ---------- Build & write -------------------------------------------------------------------------

def build(starter: bool) -> list:
    cells = [
        md(INTRO_MD),
        code(CELL_INSTALL),
        code(bootstrap_cell(
            "challenge_2_adversarial",
            [
                "challenge_2_adversarial/data/imagenet_classes.txt",
                "challenge_2_adversarial/data/images/panda.jpg",
                "challenge_2_adversarial/data/images/school_bus.jpg",
                "challenge_2_adversarial/data/images/golden_retriever.jpg",
                "challenge_2_adversarial/data/images/traffic_light.jpg",
                "challenge_2_adversarial/data/images/espresso.jpg",
            ],
        )),
        code(CELL_IMPORTS),
        md("## 1. Caricamento del modello e wrapping"),
        code(CELL_LOAD_MODEL),
        code(CELL_NORMALIZED_MODEL),
        md("## 2. Helper per immagini e predizioni"),
        code(CELL_PREPROCESS_HELPER),
        code(CELL_BASELINE_PREDICT),
        md(TODO1_MD),
        code(TODO1_STARTER if starter else TODO1_SOLUTION),
        code(CELL_VERIFY_TODO1),
        md(TODO2_MD),
        code(TODO2_STARTER if starter else TODO2_SOLUTION),
        md(TODO3_MD),
        code(TODO3_STARTER if starter else TODO3_SOLUTION),
        md("### Visualizzazione: griglia ε-sweep + curva accuracy/eps"),
        code(CELL_VIZ_GRID),
        md(TODO4_MD),
        code(TODO4_STARTER if starter else TODO4_SOLUTION),
        md(CLOSING_MD),
    ]
    if not starter:
        cells.extend([
            md(STRETCH_MD),
            code(STRETCH_S1),
            code(STRETCH_S2),
        ])
    return cells


def main():
    write(make_nb(build(starter=True)),  OUT_DIR / "starter.ipynb")
    write(make_nb(build(starter=False)), OUT_DIR / "solution.ipynb")
    print(f"Wrote {OUT_DIR / 'starter.ipynb'}")
    print(f"Wrote {OUT_DIR / 'solution.ipynb'}")


if __name__ == "__main__":
    main()
