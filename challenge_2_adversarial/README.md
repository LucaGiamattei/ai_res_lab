# Challenge 2 — Rompi il modello

**Tempo stimato:** 30 minuti.
**Articoli AI Act:** Art. 15 (Accuratezza, robustezza, cibersicurezza). Allegato IV §3.

## Obiettivo

Realizzare un attacco *Fast Gradient Sign Method* (FGSM, Goodfellow et al. 2015) su un classificatore ImageNet pre-addestrato (`MobileNetV2` da `torchvision`), misurarne l'effetto al variare di ε, e produrre un'evidence row che operazionalizzi la nozione di "robustezza" tramite un budget di perturbazione esplicito.

## Cosa imparate

- Che un modello "production-grade" pre-addestrato è suscettibile a perturbazioni **impercettibili all'occhio umano**.
- Che "robustezza" senza un valore di ε è una parola vuota: serve un budget operazionale.
- Che FGSM è il **più debole** degli attacchi di gradient-based ML — fa da baseline per discutere PGD, AutoAttack.
- Che il test produce un valore numerico (ε_max a una soglia di accuratezza data) che è una metrica auditabile.

## File

```
challenge_2_adversarial/
├── README.md
├── starter.ipynb
├── solution.ipynb
├── data/
│   ├── images/                 <- 5 immagini ImageNet, pubblico dominio
│   │   ├── README.md
│   │   ├── panda.jpg
│   │   ├── school_bus.jpg
│   │   ├── golden_retriever.jpg
│   │   ├── traffic_light.jpg
│   │   └── espresso.jpg
│   └── imagenet_classes.txt    <- 1000 etichette ImageNet
└── img/
```

## Prerequisiti

- Conoscenza di base di PyTorch (tensori, `requires_grad`, backward).
- Aver letto la slide deck, sezione *Adversarial robustness*.

## Note operative

- **CPU only.** MobileNetV2 è scelto per essere veloce su CPU (~50 ms/immagine). Non serve GPU su Colab.
- **Download dei pesi.** Al primo `MobileNet_V2_Weights.IMAGENET1K_V2`, `torchvision` scarica ~14 MB. Su Colab è tipicamente <5 secondi.
- **Reproducibilità.** Tutti i seed sono fissati a 42. Lo stesso ε produce gli stessi risultati su CPU.

## Tempo per blocco (indicativo)

| Blocco | Min |
|--------|-----|
| Setup, modello, predict baseline 5 immagini | 5 |
| Implementare `fgsm_attack()` (TODO 1) | 8 |
| Attacco singolo su panda (TODO 2) | 5 |
| Sweep ε su 5 immagini (TODO 3) | 7 |
| Compilazione evidence row (TODO 4) | 5 |

## Output attesi

1. `starter.ipynb` con tutti i TODO compilati.
2. Una figura `img/eps_sweep.png` (generata dal notebook) con la curva accuratezza vs ε.
3. Una riga nuova in `shared/evidence_template.csv` con la robustezza FGSM operazionalizzata.
