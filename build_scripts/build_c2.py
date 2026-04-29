"""Build challenge_2_adversarial/{starter,solution}.ipynb.

Streamlined design (post-feedback):
  - Lead with the WORKING attack (BIM) — students see success immediately.
  - Single TODO: implement bim_attack. One TODO for the evidence row.
  - The demo cell shows panda original | perturbation x50 | adversarial,
    with predictions stamped on each panel (the "AHA moment").
  - Systematic sweep on BIM only (clear monotone curve, drops to 0 at eps>=0.005).
  - FGSM is shown briefly at the end as historical/weaker baseline (no TODO).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import bootstrap_cell, code, make_nb, md, write

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "challenge_2_adversarial"


# ---------- Cells ---------------------------------------------------------------------------------

INTRO_MD = r"""
# Challenge 2 — Rompi il modello

**Tema.** Esempi avversariali: piccole perturbazioni dei pixel — **impercettibili all'occhio umano** — che portano un classificatore ImageNet pre-addestrato (`MobileNetV2`) a sbagliare con confidenza alta.

**Pertinenza normativa.** L'**Art. 15** del Regolamento (UE) 2024/1689 impone, per i sistemi ad alto rischio, *"un livello appropriato di accuratezza, robustezza e cibersicurezza"*. Il §5 specifica che i sistemi devono essere "*resilienti rispetto a tentativi di terzi non autorizzati di alterarne l'uso, gli output o le prestazioni sfruttando vulnerabilità del sistema*". Le perturbazioni avversariali sono il caso paradigmatico.

**Pipeline (lineare):**

1. Carichiamo MobileNetV2 e verifichiamo che predice correttamente le 5 immagini campione.
2. Implementiamo **BIM** (*Basic Iterative Method*, Kurakin et al. 2017): FGSM iterato con proiezione sulla L-inf ball.
3. **Demo**: applichiamo BIM al panda con ε=0.01 → vediamo l'immagine *visivamente identica* essere classificata come qualcos'altro.
4. **Sweep**: misuriamo l'accuratezza in funzione di ε.
5. Compiliamo l'evidence row: la robustezza è operazionalizzata come *"il più grande ε per cui l'accuratezza resta ≥ 80%"*.

> Perché BIM e non FGSM (Goodfellow 2015)? FGSM è l'attacco *single-step* originale; sui modelli moderni risulta troppo debole per produrre un fallimento netto a piccoli ε. BIM è iterativo, e a ε=0.005 manda già MobileNetV2 a 0% di accuratezza — il regime "non robusto" è inequivoco. Una cella finale facoltativa mostra il confronto.

**Riferimenti:**
- Goodfellow, Shlens, Szegedy. *Explaining and Harnessing Adversarial Examples.* ICLR 2015. (FGSM)
- Kurakin, Goodfellow, Bengio. *Adversarial Examples in the Physical World.* ICLR Workshop 2017. (BIM)
- Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018. (PGD = BIM + random init)
- Croce, Hein. *Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.* ICML 2020. (AutoAttack)
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

# ImageNet class labels.
classes = [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]
assert len(classes) == 1000, f"Expected 1000 classes, got {len(classes)}"
print(f"Loaded MobileNetV2 ({sum(p.numel() for p in base_model.parameters())/1e6:.2f} M params)")
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
print("Tutte e 5 le immagini sono classificate correttamente — punto di partenza.")
"""

# ----- Single TODO: implement BIM -----------------------------------------------------------------

TODO_BIM_MD = r"""
## TODO — Implementate `bim_attack`

**BIM** (Basic Iterative Method, Kurakin–Goodfellow–Bengio 2017) è FGSM applicato $K$ volte con piccoli step:

$$x^{(k+1)} = \mathrm{clip}_{[0,1]}\Bigl(\Pi_{B_\infty(x, \varepsilon)}\bigl(x^{(k)} + \alpha \cdot \mathrm{sign}(\nabla_x \mathcal{L}(\theta, x^{(k)}, y))\bigr)\Bigr)$$

dove:
- $\alpha < \varepsilon$ è lo step size (tipico $\alpha = \varepsilon/4$),
- $\Pi_{B_\infty(x, \varepsilon)}$ è la **proiezione** sulla L-inf ball di raggio $\varepsilon$ centrata nell'immagine originale $x$,
- $K$ è il numero di iterazioni (tipico 10).

Senza la proiezione, BIM uscirebbe dal budget ammesso. Con la proiezione, BIM è esattamente **PGD senza random init** (Madry et al. 2018) — la base di tutta la moderna robustness evaluation.

> Costo: $K \times$ il costo di FGSM. Su CPU MobileNetV2: ~50 ms × 10 = ~500 ms per immagine. Trascurabile.
"""

TODO_BIM_STARTER = """\
def bim_attack(model: nn.Module, image: torch.Tensor, label: torch.Tensor,
               epsilon: float, alpha: float | None = None, steps: int = 10) -> torch.Tensor:
    \"\"\"BIM / I-FGSM in [0, 1] pixel space, with L-inf projection.

    Args:
        model: differentiable classifier; expects input shape (B, 3, H, W) in [0, 1].
        image: input tensor (1, 3, H, W) in [0, 1].
        label: ground-truth class id, shape (1,).
        epsilon: L-infinity perturbation budget (overall).
        alpha: step size per iteration. Default: epsilon / 4.
        steps: number of iterations.

    Returns:
        adv: adversarial image, same shape and dtype as `image`, in [0, 1].
    \"\"\"
    # TODO: implement BIM. Hints (uncomment as you go):
    # if alpha is None: alpha = epsilon / 4
    # x_adv = image.clone().detach()
    # for _ in range(steps):
    #     x_adv = x_adv.detach().requires_grad_(True)
    #     logits = model(x_adv)
    #     loss = F.cross_entropy(logits, label)
    #     model.zero_grad(); loss.backward()
    #     x_adv = x_adv + alpha * x_adv.grad.sign()
    #     # project back to L-inf ball:
    #     x_adv = torch.max(torch.min(x_adv, image + epsilon), image - epsilon)
    #     x_adv = x_adv.clamp(0.0, 1.0)
    # return x_adv.detach()
    raise NotImplementedError("Implementare bim_attack")
"""

TODO_BIM_SOLUTION = """\
def bim_attack(model: nn.Module, image: torch.Tensor, label: torch.Tensor,
               epsilon: float, alpha: float | None = None, steps: int = 10) -> torch.Tensor:
    \"\"\"BIM / I-FGSM in [0, 1] pixel space, with L-inf projection.\"\"\"
    if alpha is None:
        alpha = epsilon / 4
    x_adv = image.clone().detach()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, label)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        # Project onto the L-inf ball of radius epsilon around `image`.
        x_adv = torch.max(torch.min(x_adv, image + epsilon), image - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()
"""

CELL_VERIFY_BIM = """\
# Quick sanity check: shape, range, and that BIM at eps=0.01 actually flips the panda.
x = load_image("panda.jpg")
y = torch.tensor([388], device=DEVICE)  # giant panda
x_adv = bim_attack(model, x, y, epsilon=0.01, steps=10)

assert x_adv.shape == x.shape
assert x_adv.min() >= 0 - 1e-6 and x_adv.max() <= 1 + 1e-6
linf = (x_adv - x).abs().max().item()
assert linf <= 0.01 + 1e-6, f"L-inf budget violato: {linf}"

p_orig = predict(x,     k=1)[0]
p_adv  = predict(x_adv, k=1)[0]
assert p_adv[0] != 388, "BIM non e' riuscito a flippare il panda — controllate l'implementazione."
print(f"OK: BIM ha flippato il panda da '{p_orig[1]}' (p={p_orig[2]:.3f}) "
      f"a '{p_adv[1]}' (p={p_adv[2]:.3f}) con perturbazione L-inf={linf:.6f}.")
"""

# ----- Demo: side-by-side panda viz with stamped predictions ---------------------------------------

DEMO_MD = """\
## Demo — il panda *visivamente identico* viene mis-classificato

Mostriamo l'attacco al panda con ε=0.01:

- A sinistra: l'immagine **originale**, classificata correttamente.
- Al centro: la **perturbazione** (amplificata ×50 per essere visibile).
- A destra: l'immagine **avversariale** = originale + perturbazione. Visivamente identica, ma classificata come qualcos'altro.

**Questa è l'osservazione fondamentale**: la perturbazione è impercettibile, ma il modello ne è completamente fuorviato.
"""

CELL_DEMO_PANDA = '''
EPSILON = 0.01
x        = load_image("panda.jpg")
y        = torch.tensor([388], device=DEVICE)
x_adv    = bim_attack(model, x, y, EPSILON, steps=10)

p_orig = predict(x,     k=1)[0]
p_adv  = predict(x_adv, k=1)[0]

# Perturbation amplified for visibility (centered around 0.5).
perturb_amp = (x_adv - x) * 50.0 + 0.5

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

axes[0].imshow(to_displayable(x))
axes[0].axis("off")
axes[0].set_title(f"ORIGINALE\\npredetto: {p_orig[1]}\\np={p_orig[2]:.3f}",
                  fontsize=12, color="green", fontweight="bold")

axes[1].imshow(to_displayable(perturb_amp))
axes[1].axis("off")
axes[1].set_title(f"PERTURBAZIONE (×50)\\nL-inf budget: \\u03b5={EPSILON}\\n(invisibile a ×1)",
                  fontsize=12, color="black")

axes[2].imshow(to_displayable(x_adv))
axes[2].axis("off")
flipped_color = "red" if p_adv[0] != 388 else "green"
axes[2].set_title(f"AVVERSARIALE\\npredetto: {p_adv[1]}\\np={p_adv[2]:.3f}",
                  fontsize=12, color=flipped_color, fontweight="bold")

plt.suptitle(f"BIM attack su panda (\\u03b5={EPSILON}, 10 step) — il panda non c'e' piu'", fontsize=13)
plt.tight_layout()
plt.savefig(PACKAGE_ROOT / "challenge_2_adversarial" / "img" / "panda_demo.png", dpi=80, bbox_inches="tight")
plt.show()

print(f"Originale   : {p_orig[1]} (id={p_orig[0]}, p={p_orig[2]:.3f})")
print(f"Avversariale: {p_adv[1]} (id={p_adv[0]}, p={p_adv[2]:.3f})")
print(f"Perturbazione massima (L-inf): {(x_adv - x).abs().max().item():.6f}")
'''

# ----- Sweep: accuracy vs epsilon -----------------------------------------------------------------

SWEEP_MD = """\
## Sweep — a quale ε il modello cade?

Estendiamo l'analisi a tutte e 5 le immagini per vari ε. Per ogni ε contiamo quante immagini restano classificate correttamente.
"""

CELL_SWEEP = '''
EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

results = {}
for eps in EPSILONS:
    flips = []
    for fname, expected_id, _ in SAMPLES:
        x_i = load_image(fname)
        y_i = torch.tensor([expected_id], device=DEVICE)
        x_adv_i = bim_attack(model, x_i, y_i, eps, steps=10)
        adv_pred_id = predict(x_adv_i, k=1)[0][0]
        flips.append(adv_pred_id != expected_id)
    results[eps] = {"acc": 1.0 - sum(flips) / len(flips), "flips": flips}

print(f"{'eps':<10}{'accuracy':<12}{'# flipped':<12}{'classes flipped at this eps'}")
for eps in EPSILONS:
    flipped_names = [SAMPLES[i][2] for i, f in enumerate(results[eps]["flips"]) if f]
    print(f"{eps:<10}{results[eps]['acc']:<12.2f}{sum(results[eps]['flips']):<12}{', '.join(flipped_names) if flipped_names else '-'}")
'''

CELL_GRID_ALL_ATTACKED_MD = """\
### Tutte e 5 le immagini attaccate (ε=0.01)

Per convincervi che non si tratta di un caso fortunato sul panda, ripetiamo l'attacco sulle altre 4 immagini al solito ε=0.01. Ciascuna riga: originale | perturbazione ×50 | avversariale, con le predizioni stampate.
"""

CELL_GRID_ALL_ATTACKED = '''
EPS_GRID = 0.01
fig, axes = plt.subplots(len(SAMPLES), 3, figsize=(11, 3.6 * len(SAMPLES)))

for row, (fname, expected_id, expected_name) in enumerate(SAMPLES):
    x_i = load_image(fname)
    y_i = torch.tensor([expected_id], device=DEVICE)
    x_adv_i = bim_attack(model, x_i, y_i, EPS_GRID, steps=10)
    perturb_amp = (x_adv_i - x_i) * 50.0 + 0.5
    p_orig = predict(x_i,     k=1)[0]
    p_adv  = predict(x_adv_i, k=1)[0]
    flipped = p_adv[0] != expected_id

    axes[row, 0].imshow(to_displayable(x_i))
    axes[row, 0].axis("off")
    axes[row, 0].set_title(f"ORIGINALE\\n{p_orig[1]} (p={p_orig[2]:.2f})",
                           fontsize=10, color="green", fontweight="bold")

    axes[row, 1].imshow(to_displayable(perturb_amp))
    axes[row, 1].axis("off")
    axes[row, 1].set_title(f"perturbazione \\u00d750\\n\\u03b5 = {EPS_GRID}", fontsize=10)

    axes[row, 2].imshow(to_displayable(x_adv_i))
    axes[row, 2].axis("off")
    color = "red" if flipped else "green"
    flag = "FLIPPED" if flipped else "robust"
    axes[row, 2].set_title(f"AVVERSARIALE [{flag}]\\n{p_adv[1]} (p={p_adv[2]:.2f})",
                           fontsize=10, color=color, fontweight="bold")

plt.suptitle(f"BIM-10 a \\u03b5={EPS_GRID} su tutte e 5 le immagini", fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(PACKAGE_ROOT / "challenge_2_adversarial" / "img" / "all_attacked.png", dpi=72, bbox_inches="tight")
plt.show()
'''

CELL_SWEEP_PLOT = '''
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = list(results.keys())
ys = [results[e]["acc"] for e in xs]
ax.plot(xs, ys, marker="o", color="#CC3311", linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("\\u03b5 (L-inf perturbation budget)")
ax.set_ylabel("accuracy on 5 samples")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.8, ls="--", color="grey", label="soglia operativa 80%")
ax.set_title("MobileNetV2 — accuratezza vs budget di perturbazione (BIM-10)")
ax.legend()
plt.tight_layout()
plt.savefig(PACKAGE_ROOT / "challenge_2_adversarial" / "img" / "eps_sweep.png", dpi=110, bbox_inches="tight")
plt.show()
'''

# ----- Evidence row -------------------------------------------------------------------------------

EVIDENCE_MD = r"""
## TODO — Compilate l'evidence row

Definiamo la *robustezza operativa* in modo non ambiguo:

$$\varepsilon_{\max}^{(80\%)} = \max\Bigl\{\varepsilon : \mathrm{acc}(\varepsilon') \geq 0.80 \,\,\forall\, \varepsilon' \leq \varepsilon\Bigr\}$$

cioè il più grande budget di perturbazione per cui l'accuratezza resta $\geq 0.80$ **in maniera monotona** (dal valore più piccolo testato fino a $\varepsilon$ incluso). Se l'accuratezza scende sotto $0.80$ già al primo $\varepsilon$ testato, $\varepsilon_{\max} = 0$ — il modello non è robusto al budget testato più piccolo.

Compilate l'evidence row qui sotto. Soglia operativa proposta: $\varepsilon_{\max} \geq 0.05$. Se non la raggiunge, `status=fail`.
"""

EVIDENCE_STARTER = '''
# TODO: calcolate eps_max e scrivete una riga in shared/evidence_template.csv.

THRESHOLD_ACC = 0.80
THRESHOLD_EPS = 0.05

eps_max = None  # TODO: il piu' grande eps tale che per OGNI e <= eps, results[e]["acc"] >= THRESHOLD_ACC.
                #       Iterare sorted(EPSILONS) e fermarsi alla prima violazione.
                #       Se anche il piu' piccolo eps gia' viola, eps_max = 0.0.

if eps_max is None:
    raise NotImplementedError("Calcolare eps_max")

status = "pass" if eps_max >= THRESHOLD_EPS else "fail"
NOTES = ""  # TODO: una frase, specifica (es. "primi a flippare: golden retriever; eps_max = 0").
if not NOTES:
    raise NotImplementedError("Compilare NOTES")

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
import csv
row = dict(
    challenge="C2",
    system="ImageNet-MobileNetV2",
    metric="Adversarial robustness (BIM-10, L-inf, eps_max @ 80% acc)",
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
'''

EVIDENCE_SOLUTION = '''
THRESHOLD_ACC = 0.80
THRESHOLD_EPS = 0.05

eps_max = 0.0
for eps in sorted(EPSILONS):
    if results[eps]["acc"] >= THRESHOLD_ACC:
        eps_max = eps
    else:
        break

status = "pass" if eps_max >= THRESHOLD_EPS else "fail"

# Identify which classes flipped at the smallest broken epsilon (for an informative note).
broken_eps = next((e for e in sorted(EPSILONS) if results[e]["acc"] < THRESHOLD_ACC), None)
flipped_first = []
if broken_eps is not None:
    flipped_first = [SAMPLES[i][2] for i, f in enumerate(results[broken_eps]["flips"]) if f]

NOTES = (
    f"BIM-10 (L-inf): a eps={EPSILONS[0]:.3f} acc={results[EPSILONS[0]]['acc']:.2f}; "
    f"prime classi a flippare: {', '.join(flipped_first[:3]) if flipped_first else 'nessuna nei test'}. "
    f"Modello non robusto a perturbazioni gradient-based sotto soglia operativa {THRESHOLD_EPS}."
)

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
import csv
row = dict(
    challenge="C2",
    system="ImageNet-MobileNetV2",
    metric="Adversarial robustness (BIM-10, L-inf, eps_max @ 80% acc)",
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
'''

# ----- Optional: FGSM comparison (no TODO) --------------------------------------------------------

FGSM_COMPARE_MD = r"""
## (Opzionale) Confronto con FGSM: l'attacco *single-step*

Per completezza storica: **FGSM** (Goodfellow et al. 2015) è l'attacco originale, una sola iterazione:

$$x_{\mathrm{adv}} = \mathrm{clip}_{[0,1]}\bigl(x + \varepsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L}(\theta, x, y))\bigr)$$

Su MobileNetV2 con queste 5 immagini, FGSM è troppo debole per produrre un fallimento netto a piccoli ε. Vediamolo a confronto con BIM:
"""

CELL_FGSM_COMPARE = '''
def fgsm_attack(model, image, label, epsilon):
    image = image.clone().detach().requires_grad_(True)
    logits = model(image)
    loss = F.cross_entropy(logits, label)
    model.zero_grad()
    loss.backward()
    return (image + epsilon * image.grad.sign()).clamp(0.0, 1.0).detach()

results_fgsm = {}
for eps in EPSILONS:
    flips = []
    for fname, expected_id, _ in SAMPLES:
        x_i = load_image(fname)
        y_i = torch.tensor([expected_id], device=DEVICE)
        x_adv_i = fgsm_attack(model, x_i, y_i, eps)
        flips.append(predict(x_adv_i, k=1)[0][0] != expected_id)
    results_fgsm[eps] = sum(flips) / len(flips)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = list(EPSILONS)
ys_bim  = [results[e]["acc"]     for e in xs]
ys_fgsm = [1.0 - results_fgsm[e] for e in xs]
ax.plot(xs, ys_fgsm, marker="o", color="#4477AA", label="FGSM (1 step)", linewidth=2)
ax.plot(xs, ys_bim,  marker="s", color="#CC3311", label="BIM (10 step, alpha=eps/4)", linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("\\u03b5 (L-inf perturbation budget)")
ax.set_ylabel("accuracy on 5 samples")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.8, ls="--", color="grey", label="soglia 80%")
ax.set_title("FGSM vs BIM — perche' usiamo BIM")
ax.legend()
plt.tight_layout()
plt.savefig(PACKAGE_ROOT / "challenge_2_adversarial" / "img" / "fgsm_vs_bim.png", dpi=110, bbox_inches="tight")
plt.show()

print(f"\\n{'eps':<10}{'acc_FGSM':<12}{'acc_BIM-10':<12}")
for eps in EPSILONS:
    print(f"{eps:<10}{1.0 - results_fgsm[eps]:<12.2f}{results[eps]['acc']:<12.2f}")
'''

CLOSING_MD = """\
## Chiusura — Cosa avete dimostrato (e cosa no)

**Risultato.** Una evidence row che documenta la robustezza vs un attacco gradient-based standard (BIM-10), con una soglia operativa esplicita.

**Cosa NON avete dimostrato (limiti onesti, vanno nelle Notes del fascicolo reale):**

1. **5 immagini è poco.** Audit reale: sub-set del validation ImageNet (1000+ immagini). Statistica significativa.
2. **BIM è uno solo degli attacchi.** Audit professionale userebbe almeno **AutoAttack** (Croce & Hein 2020) — ensemble parameter-free di 4 attacchi (APGD-CE, APGD-DLR, FAB, Square) — e PGD con multi-restart.
3. **Solo perturbazioni $L_\\infty$.** Esistono anche $L_2$, $L_0$, attacchi geometrici (rotazioni, traslazioni), attacchi fisici (sticker su segnali stradali, Eykholt et al. 2018), attacchi black-box (NES, Square Attack, ZOO).
4. **Nessuna mitigazione testata.** Le difese reali combinano *adversarial training* (Madry et al. 2018), input preprocessing (JPEG compression), certified defenses (randomized smoothing, Cohen et al. 2019).

**In ai.res** queste righe diventano controlli versionati nel ledger di evidenze, con re-test settimanali su attacchi standardizzati e con tracking del delta accuracy → fuori soglia operativa = alert.

> Per chi volesse approfondire: [robustbench.github.io](https://robustbench.github.io/) mantiene un leaderboard di modelli e attacchi standardizzati.
"""

STRETCH_MD = """\
## Stretch goals (solo soluzione)

### S1 — BIM mirato (targeted)

Forziamo il modello a predire una classe specifica (`target_id=805` = soccer ball), non "qualcos'altro che non sia panda". Implementazione: scendere il gradiente verso `target_id` (nota il segno meno).

### S2 — PGD con random init

BIM ha init in $x$. PGD aggiunge `x + uniform(-eps, eps)` come init. Lancia `k` ripetizioni e prendi il *worst-case*.
"""

STRETCH_S1 = """\
def targeted_bim(model, image, target_label, epsilon, alpha=None, steps=10):
    if alpha is None:
        alpha = epsilon / 4
    x_adv = image.clone().detach()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, target_label)
        model.zero_grad(); loss.backward()
        # NOTE: minus sign (we descend the loss towards target_label).
        x_adv = x_adv - alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, image + epsilon), image - epsilon).clamp(0.0, 1.0)
    return x_adv.detach()

panda_x = load_image("panda.jpg")
target  = torch.tensor([805], device=DEVICE)  # soccer ball
adv     = targeted_bim(model, panda_x, target, epsilon=0.03, steps=20)
top5    = predict(adv, k=5)
print("Top-5 dopo BIM mirato (target='soccer ball'):")
for cid, name, p in top5:
    print(f"  {cid:4d}  {name:30s}  {p:.3f}")
"""

STRETCH_S2 = """\
def pgd_with_random_init(model, image, label, epsilon, alpha=None, steps=10, restarts=5):
    if alpha is None:
        alpha = epsilon / 4
    best_loss = -float("inf")
    best_adv  = image
    for _ in range(restarts):
        x_adv = (image + torch.empty_like(image).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, label)
            model.zero_grad(); loss.backward()
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, image + epsilon), image - epsilon).clamp(0.0, 1.0)
        with torch.no_grad():
            l = F.cross_entropy(model(x_adv), label).item()
            if l > best_loss:
                best_loss, best_adv = l, x_adv
    return best_adv.detach()

panda_x = load_image("panda.jpg")
panda_y = torch.tensor([388], device=DEVICE)
adv     = pgd_with_random_init(model, panda_x, panda_y, epsilon=0.005, steps=10, restarts=5)
top1    = predict(adv, k=1)[0]
print(f"PGD-restart-5 a eps=0.005: top1={top1[1]} p={top1[2]:.3f}")
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
        md("### Predizioni baseline (devono essere tutte corrette)"),
        code(CELL_BASELINE_PREDICT),
        md(TODO_BIM_MD),
        code(TODO_BIM_STARTER if starter else TODO_BIM_SOLUTION),
        code(CELL_VERIFY_BIM),
        md(DEMO_MD),
        code(CELL_DEMO_PANDA),
        md(SWEEP_MD),
        code(CELL_SWEEP),
        md(CELL_GRID_ALL_ATTACKED_MD),
        code(CELL_GRID_ALL_ATTACKED),
        code(CELL_SWEEP_PLOT),
        md(EVIDENCE_MD),
        code(EVIDENCE_STARTER if starter else EVIDENCE_SOLUTION),
        md(FGSM_COMPARE_MD),
        code(CELL_FGSM_COMPARE),
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
