# Challenge 4 — Difese LLM: misurare che funzionano

**Tempo stimato:** 20–25 minuti.
**Articoli AI Act:** Art. 9.2.d (misure di mitigazione), Art. 15 §3 (cibersicurezza misurabile). Allegato IV §2.g.

**Prerequisito:** C3 completato, chiave Groq disponibile.

## Obiettivo

In C3 avete dimostrato che il sistema è vulnerabile. Qui dimostrate che una mitigazione **funziona** — con numeri pre/post, non con asserzioni di principio.

Pipeline lineare, senza fronzoli:

1. Caricate **5 attacchi pre-definiti** che bucano il system prompt baseline di C3.
2. Eseguite gli attacchi → un **giudice LLM** (`llama-3.3-70b-versatile`) assegna a ciascuno uno score 0–10 secondo una rubrica strutturata.
3. **TODO unico:** progettate il **vostro** system prompt indurito.
4. Eseguite gli stessi 5 attacchi contro la vostra difesa → nuovi score.
5. Confrontate. Lo *Δ score* (`mean_pre − mean_post`) è l'evidenza che la difesa riduce il rischio.

In più, una cella opzionale finale confronta la vostra difesa con un prompt indurito di riferimento (`prompts/bank_assistant_hardened.txt`).

## Cosa imparate

- Che le difese LLM **funzionano** se ben scritte — e che la prova è un numero, non un'opinione.
- Che valutare a mano N attacchi non scala: serve un **giudice automatico** con rubrica esplicita per produrre un'evidenza ripetibile (Art. 15 §3).
- Che un giudice LLM è **anche lui** vulnerabile: rubrica strutturata + output JSON + temperature 0 sono mitigazioni parziali, non eliminano il problema.
- Che in un fascicolo Annex IV una mitigazione non documentata da metriche pre/post non vale come evidenza.

## File

```
challenge_4_llm_defense/
├── README.md
├── starter.ipynb
├── solution.ipynb
└── prompts/
    ├── judge_system.txt              <- rubrica per LLM-as-judge
    ├── bank_assistant_hardened.txt   <- system prompt indurito di riferimento
    └── jailbreak_seed_prompts.txt    <- 5 jailbreak pre-definiti
```

## Output attesi

1. `starter.ipynb` con il TODO compilato (`MY_HARDENED_SYSTEM` riempito).
2. Due evidence row aggiuntive in `shared/evidence_template.csv`:
   - `C4 / BankAssistant-Llama3.1-8B / mean judge score / pre-defense / ...`
   - `C4 / BankAssistant-Llama3.1-8B-hardened / mean judge score / post-defense / ... / mitigation = "Hardened system prompt (designed by student)"`

## ⚠️ Etica e LLM-as-judge

Un LLM giudice **non è un giudice neutrale**. È soggetto a:

- **Bias di lunghezza** — risposte più lunghe ottengono più spesso punteggi alti.
- **Bias di stile** — risposte assertive vengono ritenute più affidabili anche se sbagliate.
- **Self-consistency** — chiedere lo stesso giudizio 5 volte può dare 3 esiti diversi a temperature > 0.

Per questo la rubrica forza: (a) `temperature=0`, (b) output JSON con campi puntuali, (c) la motivazione *prima* del punteggio (per ridurre l'ancoraggio sul numero finale). Sono mitigazioni parziali. In un audit reale si usano ensemble di giudici diversi + validazione umana su un sample.
