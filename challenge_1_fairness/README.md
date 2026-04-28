# Challenge 1 — Cerca il bias

**Tempo stimato:** 25 minuti.
**Articoli AI Act:** Art. 9 (Risk management), Art. 10 (Data governance), Art. 13 (Trasparenza), Art. 14 (Sorveglianza umana). Allegato IV §2.

## Obiettivo

Costruire un classificatore binario di credit scoring sul dataset *Statlog (German Credit)*, condurre un fairness audit usando `fairlearn`, applicare una mitigazione post-hoc e produrre due evidence row (baseline + mitigato).

## Cosa imparate

- Che la fairness non è una proprietà unica: ci sono **almeno quattro metriche** non equivalenti tra loro.
- Che esiste un teorema di impossibilità (Chouldechova 2017, Kleinberg et al. 2016) — non si possono soddisfare contemporaneamente *demographic parity*, *equal opportunity* e *predictive parity* se il base rate differisce tra gruppi.
- Che una mitigazione "fair" può **migliorare un parametro e peggiorarne un altro**: la scelta tra le metriche è una scelta di policy, non tecnica.
- Che l'audit produce un **artefatto strutturato** (evidence row), non un grafico in un notebook.

## File

```
challenge_1_fairness/
├── README.md
├── starter.ipynb       <- da compilare
├── solution.ipynb      <- riferimento docente, già compilato
└── img/                <- figure di riferimento
```

## Prerequisiti

- Aver letto la slide deck "AI Testing — Parte teorica", sezione *Fairness in supervised learning*.
- Familiarità con `scikit-learn` a livello base (Pipeline, train/test split, LogisticRegression).
- Aver letto `docs/EVIDENCE_TEMPLATE.md`.

## Tempo per blocco (indicativo)

| Blocco | Min |
|--------|-----|
| Caricamento dataset, mapping `personal_status` → `sex` (TODO 1) | 5 |
| Baseline LogisticRegression (TODO 2) | 5 |
| Calcolo metriche fairness (TODO 3) | 5 |
| Mitigazione con ThresholdOptimizer (TODO 4) | 5 |
| Compilazione evidence row (TODO 5) e riflessione | 5 |

## Output attesi

1. `starter.ipynb` con tutti i TODO compilati.
2. Due righe nuove in `shared/evidence_template.csv`:
    - una per il modello baseline,
    - una per il modello con `ThresholdOptimizer`.
3. Una considerazione (1 paragrafo, in markdown nel notebook) su quale metrica avete scelto come "principale" per il vostro evidence row e perché.
