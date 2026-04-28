"""Build challenge_1_fairness/{starter,solution}.ipynb.

Markdown in Italian (graduate technical register), code in English.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import bootstrap_cell, code, make_nb, md, write

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "challenge_1_fairness"


# ---------- Cells shared between starter & solution -------------------------------------------------

INTRO_MD = """\
# Challenge 1 — Cerca il bias

**Tema:** fairness audit di un classificatore di credit scoring sul dataset *Statlog (German Credit)*.

**Pertinenza normativa.** Il credit scoring rientra fra i sistemi ad alto rischio elencati nell'**Allegato III** del Regolamento (UE) 2024/1689 (AI Act), specificamente al punto 5(b): *"sistemi di IA destinati a essere utilizzati per valutare l'affidabilità creditizia delle persone fisiche o stabilirne il punteggio di credito"*. Per tali sistemi, gli articoli **9** (gestione del rischio), **10** (data governance), **13** (trasparenza) e **14** (sorveglianza umana) impongono — fra l'altro — l'esecuzione di valutazioni di fairness e la documentazione strutturata dei risultati.

**Obiettivo del laboratorio.** In ~25 minuti:

1. Carichiamo il dataset e identifichiamo un attributo protetto (sesso) — già qui emerge una sciatteria reale: il dataset codifica sesso e stato civile **insieme**, costringendoci a una scelta di proxy.
2. Addestriamo un classificatore baseline — la regressione logistica, non perché sia il modello migliore ma perché serve una baseline interpretabile per discutere fairness.
3. Misuriamo **quattro** metriche di fairness diverse (e non equivalenti) tramite `fairlearn.MetricFrame`.
4. Applichiamo una mitigazione post-hoc (`ThresholdOptimizer`) e misuriamo il tradeoff.
5. Compiliamo due **evidence row** (vedi `docs/EVIDENCE_TEMPLATE.md`) che diventano gli artefatti dell'audit.

**Riferimento normativo:** Regolamento (UE) 2024/1689, Allegato III §5(b); Artt. 9, 10, 13, 14.

**Riferimenti tecnici:**
- Hardt, Price, Srebro. *Equality of Opportunity in Supervised Learning.* NeurIPS 2016.
- Chouldechova. *Fair prediction with disparate impact.* Big Data 2017. (Teorema di impossibilità)
- Bird et al. *Fairlearn: A toolkit for assessing and improving fairness in AI.* 2020.
"""

CELL_IMPORTS_INSTALL = """\
# Install dependencies (Colab will pick these up; locally they should already be present).
%pip install -q "numpy>=1.26,<2.2" "pandas>=2.1" "scikit-learn>=1.4,<1.6" "fairlearn>=0.10" matplotlib
"""

CELL_IMPORTS = """\
import sys
import os
import warnings
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

# Detect environment
# PACKAGE_ROOT is set by the bootstrap cell above.
EVIDENCE_CSV = PACKAGE_ROOT / "shared" / "evidence_template.csv"
print(f"Evidence file: {EVIDENCE_CSV} (exists={EVIDENCE_CSV.exists()})")
"""

CELL_LOAD_DATA = """\
# Load the German Credit dataset from OpenML (no local file shipping required).
ds = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
df = ds.frame.copy()
print("Shape:", df.shape)
print("Target distribution:")
print(df["class"].value_counts(normalize=True).round(3))
df.head()
"""

CELL_FEATURES_INSPECT = """\
# Inspect features
print("Columns:", df.columns.tolist())
print()
print("personal_status values:")
print(df["personal_status"].value_counts())
"""

# TODO 1 — sex mapping
TODO1_MD = """\
## Definizione dell'attributo protetto: il *sex* derivato da `personal_status`

Il dataset German Credit codifica congiuntamente **sesso** e **stato civile** in una singola colonna `personal_status` con quattro valori. È un esempio paradigmatico di pessima data governance (Art. 10 AI Act): un attributo protetto è "nascosto" dentro una variabile composta. Per condurre un audit di fairness dobbiamo **derivare** un proxy binario `sex ∈ {male, female}`.

I valori sono:

| `personal_status` | Sesso | Stato civile |
|-------------------|-------|--------------|
| `male single` | male | single |
| `male div/sep` | male | divorziato/separato |
| `male mar/wid` | male | sposato/vedovo |
| `female div/dep/mar` | female | divorziata/separata/sposata |

> Nota: per le donne **non** abbiamo un codice "single" — un'asimmetria già in sé sospetta (cfr. Statlog dataset documentation, Hofmann 1994). Per il laboratorio assumiamo questo come proxy "good enough", ma annotate questa limitazione: in un audit reale la scriveremmo nelle *Notes* dell'evidence row.

### TODO 1 — implementate il mapping

Compilate il dizionario `PERSONAL_STATUS_TO_SEX` qui sotto e create la serie `sex`. Subito dopo c'è una cella di verifica.
"""

TODO1_STARTER = """\
# TODO 1: complete the dictionary mapping personal_status -> 'male' | 'female'.
# Use the values shown in the table above. Keys must match exactly the strings in the dataset
# (note the wrapping single quotes: e.g. "'male single'").

PERSONAL_STATUS_TO_SEX: dict[str, str] = {
    # "male single":         "male",
    # ...
}

if not PERSONAL_STATUS_TO_SEX:
    raise NotImplementedError("Compilare PERSONAL_STATUS_TO_SEX prima di proseguire.")

sex = df["personal_status"].map(PERSONAL_STATUS_TO_SEX)
print(sex.value_counts())
"""

TODO1_SOLUTION = """\
PERSONAL_STATUS_TO_SEX: dict[str, str] = {
    "male single":         "male",
    "male div/sep":        "male",
    "male mar/wid":        "male",
    "female div/dep/mar":  "female",
}

sex = df["personal_status"].map(PERSONAL_STATUS_TO_SEX)
print(sex.value_counts())
"""

CELL_VERIFY_TODO1 = """\
# Verify: the mapping must produce exactly two values and a non-degenerate split.
assert sex.notna().all(), "Mapping incompleto: alcuni valori restano NaN."
assert set(sex.unique()) == {"male", "female"}, f"Valori inattesi: {sex.unique()}"
m_share = (sex == "male").mean()
print(f"Quota male: {m_share:.3f}")
assert 0.4 < m_share < 0.85, "Distribuzione sospetta — controllate il mapping."
print("OK: mapping plausibile.")
"""

CELL_SPLIT = """\
# Drop personal_status from features (avoid leakage of the protected attribute through the proxy)
# and keep `sex` as the sensitive feature for fairness analysis.
X = df.drop(columns=["personal_status", "class"])
y = (df["class"] == "good").astype(int)  # 1 = good credit, 0 = bad

# Stratified split on target (preserves base rate)
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X, y, sex, test_size=0.25, stratify=y, random_state=SEED
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Base rate (train): {y_train.mean():.3f}")
print(f"Base rate by group (train):")
print(pd.crosstab(sex_train, y_train, normalize="index").round(3))
"""

CELL_PREPROCESS = """\
# Preprocessing scaffold: one-hot encode categoricals, standard-scale numerics.
cat_cols = X_train.select_dtypes(include="category").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()
print(f"Categorical: {len(cat_cols)} cols, Numeric: {len(num_cols)} cols")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
"""

# TODO 2 — train baseline
TODO2_MD = """\
### TODO 2 — addestrate la baseline

Costruite una `Pipeline` che concatena `preprocessor` e una `LogisticRegression(max_iter=1000, random_state=SEED)`. Addestrate su `X_train, y_train`, predicete su `X_test` e stampate accuracy / precision / recall / F1.

Subito dopo, cella di verifica: l'accuratezza deve essere > 0.7 (ragionevole per German Credit).
"""

TODO2_STARTER = """\
# TODO 2: build a sklearn Pipeline(preprocessor, LogisticRegression) and fit it.
# Then predict on the test set and report accuracy, precision, recall, F1.

baseline_clf = None  # TODO

if baseline_clf is None:
    raise NotImplementedError("Costruire e addestrare baseline_clf.")

y_pred_baseline = baseline_clf.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_baseline):.3f}")
print(f"F1:        {f1_score(y_test, y_pred_baseline):.3f}")
"""

TODO2_SOLUTION = """\
baseline_clf = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
])
baseline_clf.fit(X_train, y_train)

y_pred_baseline = baseline_clf.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_baseline):.3f}")
print(f"F1:        {f1_score(y_test, y_pred_baseline):.3f}")
"""

CELL_VERIFY_TODO2 = """\
# Verify: a sane baseline should beat 0.7 accuracy on German Credit.
acc = accuracy_score(y_test, y_pred_baseline)
assert acc > 0.7, f"Accuratezza troppo bassa ({acc:.3f}); controllate il preprocessor / pipeline."
print(f"OK: baseline accuracy = {acc:.3f}")
"""

# TODO 3 — fairness metrics
TODO3_MD = r"""
## Le quattro metriche di fairness

Definiamo $\hat{Y}$ la previsione binaria, $Y$ il target e $A$ l'attributo protetto (gruppo). Le metriche più comuni:

1. **Selection rate per gruppo** — $\Pr(\hat{Y}=1 | A=a)$.
2. **Demographic parity difference (DPD)** — $|\Pr(\hat{Y}=1|A=0) - \Pr(\hat{Y}=1|A=1)|$. Misura se i due gruppi sono "selezionati" allo stesso tasso, *a prescindere dal target*.
3. **Equal opportunity difference (EOD)** — differenza di TPR tra gruppi, $|\Pr(\hat{Y}=1|Y=1, A=0) - \Pr(\hat{Y}=1|Y=1, A=1)|$. Misura se i positivi reali ricevono predizione positiva allo stesso tasso (Hardt et al. 2016).
4. **Equalized odds difference** — il massimo tra differenza di TPR e differenza di FPR (più stringente di EOD).

> **Teorema di impossibilità (Chouldechova 2017, Kleinberg et al. 2016).** Se i base rate $\Pr(Y=1|A=a)$ differiscono fra gruppi, *non* si possono soddisfare contemporaneamente *demographic parity*, *equal opportunity* e *predictive parity*. La scelta della metrica è una **scelta di policy**, non tecnica.

### TODO 3 — calcolate le metriche

Costruite un `MetricFrame` con `selection_rate`, `true_positive_rate`, `false_positive_rate`, raggruppato per `sex_test`. Calcolate poi DPD ed equalized odds difference. Stampate tutto come tabella.
"""

TODO3_STARTER = """\
# TODO 3: build a MetricFrame and compute the fairness metrics on the baseline predictions.

metrics_baseline = None       # TODO: MetricFrame
dpd_baseline = None           # TODO: demographic_parity_difference(...)
eod_baseline = None           # TODO: equalized_odds_difference(...)

if metrics_baseline is None or dpd_baseline is None or eod_baseline is None:
    raise NotImplementedError("Compilare il calcolo delle metriche di fairness.")

print("By group (selection_rate, TPR, FPR):")
print(metrics_baseline.by_group.round(3))
print()
print(f"Demographic parity difference: {dpd_baseline:.3f}")
print(f"Equalized odds difference:     {eod_baseline:.3f}")
"""

TODO3_SOLUTION = """\
metrics_baseline = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    },
    y_true=y_test,
    y_pred=y_pred_baseline,
    sensitive_features=sex_test,
)
dpd_baseline = demographic_parity_difference(y_test, y_pred_baseline, sensitive_features=sex_test)
eod_baseline = equalized_odds_difference(y_test, y_pred_baseline, sensitive_features=sex_test)

print("By group (selection_rate, TPR, FPR):")
print(metrics_baseline.by_group.round(3))
print()
print(f"Demographic parity difference: {dpd_baseline:.3f}")
print(f"Equalized odds difference:     {eod_baseline:.3f}")
"""

CELL_VERIFY_TODO3 = """\
# Verify: numbers must be present and bounded in [0, 1].
assert metrics_baseline is not None
assert 0.0 <= abs(dpd_baseline) <= 1.0, f"DPD fuori range: {dpd_baseline}"
assert 0.0 <= abs(eod_baseline) <= 1.0, f"EOD fuori range: {eod_baseline}"
print("OK: metriche calcolate.")
"""

INTERPRET_MD = """\
### Riflessione (1 minuto, in coppia)

Osservate la tabella `by_group`. Quali domande vi vengono in mente?

- Quale gruppo è favorito dalla baseline (selection_rate più alto)?
- Il *gap* è grande o piccolo? Quanto è grande in valore assoluto? In termini relativi?
- Il base rate dei due gruppi è uguale? (Lo abbiamo stampato in fase di split.) Se non lo è, ricordate il teorema di impossibilità: una metrica può essere "soddisfatta" solo a costo di un'altra.
- Per un sistema di credit scoring reale, quale soglia regolatoria scegliereste? `0.05`? `0.10`? `0.20`? La scelta non è tecnica.
"""

# TODO 4 — mitigation
TODO4_MD = """\
## Mitigazione: `ThresholdOptimizer`

`fairlearn.postprocessing.ThresholdOptimizer` è una mitigazione **post-hoc**: prende un classificatore già addestrato e cerca soglie diverse per ciascun gruppo, ottimizzando un obiettivo (es. `accuracy`) sotto un vincolo di fairness (es. `equalized_odds`). Non riaddestra il modello — modifica solo il decision threshold.

**Pro:** non richiede di toccare il training; rapida; riproducibile.
**Contro:** richiede l'attributo protetto **a inference time** (problematico se non disponibile o se il suo uso è proibito); può ridurre l'accuratezza globale; non risolve bias strutturali del modello.

### TODO 4 — applicate la mitigazione

1. Istanziate `ThresholdOptimizer(estimator=baseline_clf, constraints='equalized_odds', objective='accuracy_score', prefit=True)`.
2. Chiamate `.fit(X_train, y_train, sensitive_features=sex_train)`.
3. Predicete su `X_test` con `sensitive_features=sex_test`. Salvate il risultato come `y_pred_mitigated`.
4. Ricompilate `metrics_mitigated`, `dpd_mitigated`, `eod_mitigated` analoghi a TODO 3.
"""

TODO4_STARTER = """\
# TODO 4: fit a ThresholdOptimizer and re-compute the fairness metrics.

mitigator = None              # TODO: ThresholdOptimizer(...)
y_pred_mitigated = None       # TODO: mitigator.predict(X_test, sensitive_features=sex_test)

metrics_mitigated = None      # TODO
dpd_mitigated = None          # TODO
eod_mitigated = None          # TODO

if mitigator is None or y_pred_mitigated is None:
    raise NotImplementedError("Implementare la mitigazione e ri-calcolare le metriche.")

acc_mit = accuracy_score(y_test, y_pred_mitigated)
print(f"Accuracy (mitigated): {acc_mit:.3f}")
print()
print("By group (mitigated):")
print(metrics_mitigated.by_group.round(3))
print()
print(f"Demographic parity difference (mitigated): {dpd_mitigated:.3f}")
print(f"Equalized odds difference (mitigated):     {eod_mitigated:.3f}")
"""

TODO4_SOLUTION = """\
mitigator = ThresholdOptimizer(
    estimator=baseline_clf,
    constraints="equalized_odds",
    objective="accuracy_score",
    prefit=True,
)
mitigator.fit(X_train, y_train, sensitive_features=sex_train)
y_pred_mitigated = mitigator.predict(X_test, sensitive_features=sex_test, random_state=SEED)

metrics_mitigated = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    },
    y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=sex_test,
)
dpd_mitigated = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sex_test)
eod_mitigated = equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=sex_test)

acc_mit = accuracy_score(y_test, y_pred_mitigated)
print(f"Accuracy (mitigated): {acc_mit:.3f}")
print()
print("By group (mitigated):")
print(metrics_mitigated.by_group.round(3))
print()
print(f"Demographic parity difference (mitigated): {dpd_mitigated:.3f}")
print(f"Equalized odds difference (mitigated):     {eod_mitigated:.3f}")
"""

CELL_COMPARE_PLOT = """\
# Comparison plot: 4 metrics, baseline vs mitigated.
labels = ["accuracy", "DPD", "EOD", "selection rate gap"]

acc_base = accuracy_score(y_test, y_pred_baseline)
acc_mit = accuracy_score(y_test, y_pred_mitigated)
sr_gap_base = (
    metrics_baseline.by_group["selection_rate"].max()
    - metrics_baseline.by_group["selection_rate"].min()
)
sr_gap_mit = (
    metrics_mitigated.by_group["selection_rate"].max()
    - metrics_mitigated.by_group["selection_rate"].min()
)

baseline_vals  = [acc_base, abs(dpd_baseline), abs(eod_baseline), sr_gap_base]
mitigated_vals = [acc_mit,  abs(dpd_mitigated), abs(eod_mitigated), sr_gap_mit]

x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, baseline_vals,  w, label="Baseline",  color="#4477AA")
ax.bar(x + w/2, mitigated_vals, w, label="Mitigated", color="#EE6677")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Value (lower = better for fairness; higher = better for accuracy)")
ax.set_title("Baseline vs Mitigated — German Credit fairness audit")
ax.legend()
for i, (b, m) in enumerate(zip(baseline_vals, mitigated_vals)):
    ax.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, m + 0.01, f"{m:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.show()
"""

# TODO 5 — evidence rows
TODO5_MD = """\
## TODO 5 — Compilate due evidence row e salvate in CSV

Aprite `docs/EVIDENCE_TEMPLATE.md` per riguardare il formato. Dovete scrivere **due righe** in `shared/evidence_template.csv`:

1. **Baseline** (`mitigation = "none"`)
2. **Mitigated** (`mitigation = "ThresholdOptimizer-EqualizedOdds"`)

Per ciascuna riga scegliete:

- **`metric`**: una metrica fra le quattro misurate. Scegliete *prima* quale sia la metrica principale per il vostro audit (e motivate la scelta nelle Notes).
- **`threshold`**: una soglia numerica; tipico `<= 0.10` per DPD/EOD.
- **`status`**: `pass` se osservato ≤ soglia, `fail` se sopra, `partial` solo se la metrica è multidimensionale.
- **`notes`**: una frase, specifica.

Il codice qui sotto fa il salvataggio. Voi compilate i campi.
"""

TODO5_STARTER = """\
# TODO 5: compilare le due evidence row e appenderle al CSV condiviso.

CHOSEN_METRIC = "Demographic parity difference"   # o EOD, o selection_rate gap, ecc.
THRESHOLD = "<= 0.10"                              # convenzione testuale

baseline_observed = abs(dpd_baseline)              # adattate se cambiate metrica
mitigated_observed = abs(dpd_mitigated)
baseline_status   = "pass" if baseline_observed   <= 0.10 else "fail"
mitigated_status  = "pass" if mitigated_observed  <= 0.10 else "fail"

# TODO: scrivete una nota specifica per ciascuna riga (1 frase).
NOTES_BASELINE  = ""    # es. "Modello favorisce il gruppo male; gap oltre soglia"
NOTES_MITIGATED = ""    # es. "Mitigazione riduce il gap a costo di -X p.p. accuracy"

if not NOTES_BASELINE or not NOTES_MITIGATED:
    raise NotImplementedError("Compilate NOTES_BASELINE e NOTES_MITIGATED.")

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

rows = [
    dict(
        challenge="C1", system="GermanCredit-LogReg",
        metric=CHOSEN_METRIC, threshold=THRESHOLD,
        observed=f"{baseline_observed:.3f}", status=baseline_status,
        mitigation="none", notes=NOTES_BASELINE, timestamp=ts,
    ),
    dict(
        challenge="C1", system="GermanCredit-LogReg",
        metric=CHOSEN_METRIC, threshold=THRESHOLD,
        observed=f"{mitigated_observed:.3f}", status=mitigated_status,
        mitigation="ThresholdOptimizer-EqualizedOdds", notes=NOTES_MITIGATED, timestamp=ts,
    ),
]

import csv
EVIDENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
write_header = not EVIDENCE_CSV.exists() or EVIDENCE_CSV.stat().st_size == 0
with EVIDENCE_CSV.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if write_header:
        w.writeheader()
    for row in rows:
        w.writerow(row)

print(f"Scritte {len(rows)} righe in {EVIDENCE_CSV}")
"""

TODO5_SOLUTION = """\
CHOSEN_METRIC = "Demographic parity difference"
THRESHOLD = "<= 0.10"

baseline_observed = abs(dpd_baseline)
mitigated_observed = abs(dpd_mitigated)
baseline_status   = "pass" if baseline_observed   <= 0.10 else "fail"
mitigated_status  = "pass" if mitigated_observed  <= 0.10 else "fail"

NOTES_BASELINE  = (
    f"Baseline LogReg: gap di selection rate {abs(dpd_baseline):.3f}; "
    f"il modello favorisce il gruppo con base-rate piu' alto; metrica scelta = DPD perche' "
    f"in credit scoring l'attenzione regolatoria e' sul tasso di erogazione."
)
NOTES_MITIGATED = (
    f"ThresholdOptimizer (constr=eq_odds): DPD {abs(dpd_baseline):.3f} -> {abs(dpd_mitigated):.3f}; "
    f"costo accuracy {accuracy_score(y_test, y_pred_baseline):.3f} -> {accuracy_score(y_test, y_pred_mitigated):.3f}."
)

ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

rows = [
    dict(
        challenge="C1", system="GermanCredit-LogReg",
        metric=CHOSEN_METRIC, threshold=THRESHOLD,
        observed=f"{baseline_observed:.3f}", status=baseline_status,
        mitigation="none", notes=NOTES_BASELINE, timestamp=ts,
    ),
    dict(
        challenge="C1", system="GermanCredit-LogReg",
        metric=CHOSEN_METRIC, threshold=THRESHOLD,
        observed=f"{mitigated_observed:.3f}", status=mitigated_status,
        mitigation="ThresholdOptimizer-EqualizedOdds", notes=NOTES_MITIGATED, timestamp=ts,
    ),
]

import csv
EVIDENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
write_header = not EVIDENCE_CSV.exists() or EVIDENCE_CSV.stat().st_size == 0
with EVIDENCE_CSV.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if write_header:
        w.writeheader()
    for row in rows:
        w.writerow(row)

print(f"Scritte {len(rows)} righe in {EVIDENCE_CSV}")
print()
print("Contenuto attuale:")
print(EVIDENCE_CSV.read_text())
"""

CLOSING_MD = """\
## Chiusura — Mappa AI Act

Quello che avete appena fatto produce evidenze direttamente ascrivibili a (vedi `docs/AI_ACT_MAPPING.md`):

- **Art. 10** (Data governance) — la stessa identificazione del proxy `sex` da `personal_status` è un'evidenza di esame del dataset per individuare distorsioni.
- **Art. 9** (Risk management) — la mitigazione (`ThresholdOptimizer`) è una misura tecnica che riduce un rischio identificato.
- **Art. 13** (Trasparenza) — le note nelle vostre evidence row diventano parte delle informazioni da fornire al deployer.
- **Art. 14** (Sorveglianza umana) — la scelta della metrica e della soglia *non* va automatizzata: è una scelta di policy che richiede revisione umana.

In un sistema reale, questo audit andrebbe ripetuto periodicamente (es. settimanalmente, su slice di traffico recenti) e i risultati archiviati in modo immutabile nel ledger di evidenze.
"""

STRETCH_MD = """\
## Stretch goals (solo soluzione)

Tre direzioni per chi finisce in anticipo o per esercizio personale:

### S1 — Cambiate l'attributo protetto

Anziché `sex`, usate `age >= 25` come attributo protetto. Spesso il bias età è sostanziale in scoring (cfr. *fair lending laws* USA, 12 CFR §1002.6). Ricomputate le quattro metriche.

### S2 — Cambiate il modello

Sostituite `LogisticRegression` con `RandomForestClassifier(n_estimators=200, random_state=SEED)`. Osservate come cambia il *fairness profile*: spesso modelli più espressivi hanno bias maggiore (la *capacity* permette di adattarsi meglio a pattern correlati con l'attributo protetto).

### S3 — Discutete il caso limite del Reweighing

Non lo implementiamo, ma riflettete: il Reweighing (Kamiran & Calders 2012) modifica i *pesi* dei sample in training. Funziona solo se l'attributo protetto è disponibile in training; il caso ipotetico — *non averlo nemmeno in training* — è il problema *fairness through unawareness*, dimostrato non funzionante (Dwork et al. 2012).
"""

STRETCH_S1 = """\
# Stretch S1: change the protected attribute to age >= 25.
age_proxy = (df["age"] >= 25).astype(int).map({1: ">=25", 0: "<25"})
age_train = age_proxy.loc[X_train.index]
age_test  = age_proxy.loc[X_test.index]

mf_age = MetricFrame(
    metrics={"selection_rate": selection_rate, "true_positive_rate": true_positive_rate},
    y_true=y_test, y_pred=y_pred_baseline, sensitive_features=age_test,
)
print("Fairness by age >= 25 (baseline LogReg):")
print(mf_age.by_group.round(3))
print(f"DPD by age: {demographic_parity_difference(y_test, y_pred_baseline, sensitive_features=age_test):.3f}")
"""

STRETCH_S2 = """\
# Stretch S2: same audit with RandomForest.
rf_clf = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED)),
])
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

mf_rf = MetricFrame(
    metrics={"selection_rate": selection_rate, "true_positive_rate": true_positive_rate, "false_positive_rate": false_positive_rate},
    y_true=y_test, y_pred=y_pred_rf, sensitive_features=sex_test,
)
print("Random Forest — by group:")
print(mf_rf.by_group.round(3))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"DPD: {demographic_parity_difference(y_test, y_pred_rf, sensitive_features=sex_test):.3f}")
print(f"EOD: {equalized_odds_difference(y_test, y_pred_rf, sensitive_features=sex_test):.3f}")
"""


# ---------- Build & write -------------------------------------------------------------------------

def build(starter: bool) -> list:
    cells = [
        md(INTRO_MD),
        code(CELL_IMPORTS_INSTALL),
        code(bootstrap_cell("challenge_1_fairness", ["shared/evidence_template.csv"])),
        code(CELL_IMPORTS),
        md("## 1. Caricamento dataset"),
        code(CELL_LOAD_DATA),
        code(CELL_FEATURES_INSPECT),
        md(TODO1_MD),
        code(TODO1_STARTER if starter else TODO1_SOLUTION),
        code(CELL_VERIFY_TODO1),
        md("## 2. Train / test split e preprocessing"),
        code(CELL_SPLIT),
        code(CELL_PREPROCESS),
        md(TODO2_MD),
        code(TODO2_STARTER if starter else TODO2_SOLUTION),
        code(CELL_VERIFY_TODO2),
        md(TODO3_MD),
        code(TODO3_STARTER if starter else TODO3_SOLUTION),
        code(CELL_VERIFY_TODO3),
        md(INTERPRET_MD),
        md(TODO4_MD),
        code(TODO4_STARTER if starter else TODO4_SOLUTION),
        md("### Confronto visivo"),
        code(CELL_COMPARE_PLOT),
        md(TODO5_MD),
        code(TODO5_STARTER if starter else TODO5_SOLUTION),
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
