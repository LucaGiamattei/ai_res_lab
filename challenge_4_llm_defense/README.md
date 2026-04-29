# Challenge 4 — Difese LLM e Leaderboard

**Tempo stimato:** 25 minuti.
**Articoli AI Act:** Art. 9 (Risk management — *misure di mitigazione*), Art. 15 (Cibersicurezza). Allegato IV §2.g.

**Prerequisito:** aver completato C3 (red-team baseline). C4 estende C3 con due esperimenti che spostano la discussione dall'*identificare* il rischio al *misurarne la mitigazione*.

## Obiettivo

Trasformare i risultati di red-team di C3 (evidenze di vulnerabilità) in azioni concrete:

1. **Subtask D — Leaderboard di jailbreak con LLM-as-judge.** Un *secondo* LLM funge da giudice automatico: legge prompt + risposta e produce uno score [0–10] con motivazione, secondo una rubrica esplicita. Si fa una piccola gara fra team — chi ottiene il jailbreak con score più alto vince. Trasforma il red-team da attività narrativa a *misura ripetibile*.
2. **Subtask E — Defense-design e re-test.** Gli studenti propongono un *system prompt indurito* (o un output filter) e ri-eseguono il proprio jailbreak migliore + le 2 email di prompt injection di C3. Misurano il **delta** di robustezza e producono due evidence row (pre-defense, post-defense) — l'analogo di C1 per la fairness, ma per la sicurezza LLM.

## Cosa imparate

- Che la valutazione manuale del red-team **non scala**: serve un giudice automatico con rubrica esplicita perché il risultato sia auditabile (Art. 15 §3 — "*misure verificabili e proporzionate*").
- Che un giudice LLM è **anche lui** vulnerabile: se l'attacker controlla parte dell'input del judge (es. la risposta dell'assistente vittima), può far inquinare il punteggio. Mitigazione: rubrica con campi strutturati e output JSON validato.
- Che la mitigazione è efficace **solo se misurata**: una system prompt "più severa" potrebbe non funzionare. Il delta tra pre- e post-defense è l'evidenza.
- Che in un fascicolo Annex IV una mitigazione non documentata da una metrica pre/post non vale.

## File

```
challenge_4_llm_defense/
├── README.md
├── starter.ipynb
├── solution.ipynb
└── prompts/
    ├── judge_system.txt              <- rubrica per LLM-as-judge
    ├── bank_assistant_hardened.txt   <- system prompt indurito di riferimento
    └── jailbreak_seed_prompts.txt    <- 5 jailbreak iniziali per popolare la leaderboard
```

## Sottosfide

| Subtask | Attività | Tempo |
|---------|----------|-------|
| D       | Leaderboard di jailbreak (LLM-as-judge) | 12 min |
| E       | Defense-design + re-test (delta evidence row) | 13 min |

## Output attesi

1. `starter.ipynb` con tutti i TODO compilati.
2. Una tabella *leaderboard* con N tentativi, score giudice, motivazione.
3. Due evidence row aggiuntive in `shared/evidence_template.csv`:
   - `C4 / BankAssistant-Llama3.1-8B / Jailbreak resilience (judge score) / pre-defense / ...`
   - `C4 / BankAssistant-Llama3.1-8B-hardened / Jailbreak resilience (judge score) / post-defense / ... / mitigation = "Hardened system prompt"`

## ⚠️ Etica e LLM-as-judge

Un LLM giudice **non è un giudice neutrale**. È a sua volta soggetto a:

- **Bias di lunghezza** — risposte più lunghe ottengono più spesso punteggi alti.
- **Bias di stile** — risposte assertive vengono ritenute più affidabili anche se sbagliate.
- **Self-consistency** — chiedere lo stesso giudizio 5 volte può dare 3 esiti diversi a temperature > 0.

Per questo nella nostra rubrica forziamo: (a) `temperature=0`, (b) output JSON con campi puntuali, (c) la motivazione *prima* del punteggio (per ridurre il bias di ancoraggio sul numero finale). Sono mitigazioni parziali. In un audit reale si userebbero ensemble di giudici diversi e/o validazione umana su un sample.
