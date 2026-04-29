# Lab di AI Testing & Governance

Pacchetto laboratorio per la lezione magistrale (4 ore) su testing e governance dei sistemi di intelligenza artificiale, rivolto agli studenti del corso di laurea magistrale in Ingegneria Informatica dell'Università degli Studi di Napoli Federico II.


---

## 1. Cosa contiene questo pacchetto

Quattro sfide pratiche, ispirate a casi reali di non-conformità ai requisiti dell'AI Act (Regolamento UE 2024/1689):

| # | Sfida | Tecnica | Articoli AI Act rilevanti |
|---|-------|---------|---------------------------|
| C1 | **Cerca il bias** — fairness audit su German Credit | `fairlearn.MetricFrame`, `ThresholdOptimizer` | Art. 9, 10, 13, 14 |
| C2 | **Rompi il modello** — esempi avversariali FGSM | `torch`, MobileNetV2 pre-addestrato | Art. 15 |
| C3 | **Red-team un LLM** — jailbreak / prompt injection / hallucination | API Groq, `llama-3.1-8b-instant` | Art. 9, 13, 15 |
| C4 | **Difese LLM e leaderboard** — LLM-as-judge + defense delta | API Groq, `llama-3.3-70b-versatile` come judge | Art. 9.2.d, 15 |

Ogni sfida produce una o più **evidence row** (vedi `docs/EVIDENCE_TEMPLATE.md`): righe strutturate in formato CSV che documentano metrica, soglia, valore osservato, stato di conformità ed eventuale mitigazione. Le evidence row di tutte e tre le sfide confluiscono in `shared/evidence_template.csv`, simulando un *evidence pack* destinato al fascicolo di conformità tecnica (Art. 11, Allegato IV).

```
ai_res_lab/
├── README.md                              <- questo file
├── requirements.txt
├── verify_setup.py                        <- script di validazione end-to-end
├── docs/
│   ├── EVIDENCE_TEMPLATE.md
│   └── AI_ACT_MAPPING.md
├── shared/
│   └── evidence_template.csv
├── challenge_1_fairness/
├── challenge_2_adversarial/
├── challenge_3_llm_redteam/
└── challenge_4_llm_defense/
```

## 2. Setup

### 2.1 Google Colab (raccomandato)

Nessun setup locale e **nessun upload manuale degli asset**. Aprire il notebook `starter.ipynb` di ogni sfida direttamente su Colab:

1. `File → Upload notebook` (oppure `File → Open notebook → GitHub` con l'URL del repo).
2. La prima cella installa le dipendenze.
3. La seconda cella (**bootstrap**) rileva l'ambiente: se sta girando su Colab e non vede il pacchetto, **clona il repo pubblico** che contiene il pacchetto e fa `chdir` nella cartella della sfida — così ogni asset (prompts, immagini, classi ImageNet, CSV) è disponibile.
4. Per la sfida C3, occorre una chiave API Groq (vedi sezione 4).

Il bootstrap clona da: <https://github.com/LucaGiamattei/ai_res_lab.git>, branch `main`. Per cambiare il repo (fork interno, mirror, ecc.) modificare `REPO_URL` nella cella di bootstrap di ogni starter (3 occorrenze, una per challenge).

**Aprire direttamente da GitHub:** sostituire `github.com` con `colab.research.google.com/github` nell'URL del notebook su GitHub. Es. <https://colab.research.google.com/github/LucaGiamattei/ai_res_lab/blob/main/challenge_1_fairness/starter.ipynb>.

> Colab Free Tier è sufficiente per tutte le sfide. La C2 usa `MobileNetV2` su CPU (~50 ms/immagine); non serve GPU.

### 2.2 Locale (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Si raccomanda Python 3.11. Versioni 3.10–3.12 dovrebbero funzionare.

### 2.3 Locale (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## 3. Come usare

Ogni sfida è indipendente. L'ordine consigliato è C1 → C2 → C3, ma le sfide possono essere svolte in parallelo da team diversi.

| Sfida | Quando iniziare | Durata stimata | Cosa consegnare |
|-------|-----------------|----------------|-----------------|
| C1 | Subito dopo l'introduzione (lezione, slide 1–25) | 25 min | Notebook compilato + evidence row in `shared/evidence_template.csv` |
| C2 | A metà laboratorio | 30 min | Notebook compilato + figura comparativa originale/perturbazione/avversariale + evidence row |
| C3 | Inizio ultima ora | 35 min | Notebook compilato + evidence row per ciascuna delle 3 categorie di attacco |
| C4 | Subito dopo C3 (riusa la chiave Groq) | 25 min | Leaderboard con 3 vostri tentativi + 2 evidence row (pre/post defense) |

Ogni notebook è strutturato con celle di **TODO** marcate da `# TODO:` e da una cella markdown precedente che spiega cosa fare e perché. Dopo ogni TODO maggiore c'è una cella di verifica che stampa `expected vs actual` o asserisce un invariante: vi consente di sapere se siete sulla strada giusta senza dover eseguire l'intero notebook.

## 4. API key per la Sfida 3

La sfida C3 richiede una chiave Groq (free tier). Procedura (~30 secondi):

1. Aprire <https://console.groq.com/keys> e accedere (Google / GitHub).
2. Cliccare `Create API Key` e dare un nome (es. `lab-federico-ii`).
3. Copiare la chiave (formato `gsk_...`). **Non condividetela**: i limiti del free tier sono per-account.

Modi per fornire la chiave al notebook (in ordine di preferenza):

- **Colab (raccomandato):** `Strumenti → Secret → +Aggiungi nuovo Secret` → nome `GROQ_API_KEY`, abilitarlo per il notebook corrente.
- **Locale, env var:** `export GROQ_API_KEY=gsk_...` prima di avviare Jupyter.
- **Fallback:** la prima cella del notebook chiama `getpass.getpass()` se non trova la chiave: incollatela quando viene chiesta. La chiave non viene mai stampata né salvata.

> ⚠️ **Mai** scrivere la chiave nel codice del notebook o committarla in git.

## 5. Risoluzione problemi

| Errore | Causa probabile | Soluzione |
|--------|-----------------|-----------|
| `ModuleNotFoundError: fairlearn` | Prima cella non eseguita | Rieseguire la prima cella del notebook |
| `RuntimeError: Couldn't download MobileNet weights` | Rete bloccata o timeout | Riprovare; in Colab è quasi sempre transitorio |
| `groq.RateLimitError` | Free tier ~30 RPM, 14.4K RPD | Attendere 60 s; per più volume passare a `llama-3.3-70b-versatile` non aiuta (stesso limite) |
| `groq.AuthenticationError` | Chiave assente o errata | Verificare il Secret/env var; controllare che la chiave inizi con `gsk_` |
| `numpy.dtype size changed` o simili | Versione numpy incompatibile (Colab a volte aggiorna mid-session) | Riavviare il runtime (`Runtime → Restart`) |
| `ipywidgets`/output bloccato in Colab | Cella troppo grande | Suddividere; la C2 ha celle pesate intenzionalmente |

## 6. Per il docente

- **Notebook di soluzione:** in ogni cartella `challenge_*/` è presente un `solution.ipynb` con tutti i TODO compilati e con sezioni "Stretch goals" aggiuntive. **Non distribuire agli studenti** prima del laboratorio.
- **Cheatsheet C3:** `challenge_3_llm_redteam/solution_cheatsheet.md` documenta i pattern di jailbreak che ci si aspetta funzionino e quelli che ci si aspetta vengano rifiutati. Da consultare per validare che i risultati degli studenti siano nel range atteso.
- **Validazione end-to-end:** eseguire `python verify_setup.py` (con `GROQ_API_KEY` esportata se si vuole testare anche C3 live). Stampa un report PASS/FAIL.
- **Mappa AI Act:** `docs/AI_ACT_MAPPING.md` collega ciascun subtask agli articoli di riferimento, utile per la sessione di debrief.

## 7. Crediti e riferimenti

Riferimenti principali (citati nei notebook):

- Goodfellow, Shlens, Szegedy. *Explaining and Harnessing Adversarial Examples.* ICLR 2015. (FGSM)
- Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018. (PGD)
- Croce, Hein. *Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.* ICML 2020. (AutoAttack)
- Hardt, Price, Srebro. *Equality of Opportunity in Supervised Learning.* NeurIPS 2016.
- Chouldechova. *Fair prediction with disparate impact.* Big Data 2017. (Impossibility result)
- Greshake et al. *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* AISec 2023.
- OWASP. *Top 10 for LLM Applications.* 2025.
- AI Act — Regolamento (UE) 2024/1689.

Dataset:
- *Statlog (German Credit).* UCI Machine Learning Repository, via `sklearn.datasets.fetch_openml('credit-g')`.
- ImageNet (immagini campione di pubblico dominio da Wikimedia Commons; vedi `challenge_2_adversarial/data/images/README.md`).

Modelli LLM tramite [Groq](https://groq.com/) free tier (`llama-3.1-8b-instant`).

---

*Versione del pacchetto: 1.0 — aprile 2026. Pubblicato sotto licenza didattica per uso interno UniNa.*
