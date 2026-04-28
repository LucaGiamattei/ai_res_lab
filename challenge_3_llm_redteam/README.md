# Challenge 3 — Red-team un LLM

**Tempo stimato:** 35 minuti.
**Articoli AI Act:** Art. 9 (Risk management), Art. 13 (Trasparenza), Art. 15 (Cibersicurezza).

## Obiettivo

Condurre un red-team su un assistente bancario costruito attorno a un LLM open-weights (`llama-3.1-8b-instant` via API Groq, free tier), con tre vettori d'attacco: **direct jailbreak**, **indirect prompt injection**, **hallucination**. Produrre tre evidence row distinte, una per categoria.

## Cosa imparate

- Che un *system prompt*, anche scritto bene, è una **policy soft** — facilmente aggirata.
- Che la **prompt injection indiretta** è il vettore d'attacco principale e attualmente non risolto per gli LLM integrati con dati esterni (email, web, customer messages).
- Che la **hallucination** non è un bug — è una proprietà statistica del decoding, e il rifiuto del modello non equivale a affidabilità informativa.
- Che il red-team produce un artefatto strutturato (evidence row): *N successi su M tentativi*, soglia esplicita, classificazione dell'esito.

## File

```
challenge_3_llm_redteam/
├── README.md
├── starter.ipynb
├── solution.ipynb
├── prompts/
│   ├── bank_assistant_system.txt
│   ├── injection_email_1.txt
│   └── injection_email_2.txt
└── solution_cheatsheet.md     <- istruttore: pattern noti
```

## Prerequisiti

1. **Account Groq.** Free tier — 30 secondi su <https://console.groq.com/keys>. Dettagli nel README di livello superiore.
2. **Chiave API esportata** come `GROQ_API_KEY` (env var, Colab Secret, o paste-when-prompted).
3. Aver letto `docs/EVIDENCE_TEMPLATE.md` e la slide deck, sezione *LLM red-teaming*.

## Vincoli di rate

- `llama-3.1-8b-instant` (default): ~30 RPM, ~6K TPM, ~14.4K RPD. Comodo per il laboratorio.
- `llama-3.3-70b-versatile` (stretch): stessi limiti — non è una via d'uscita per studenti che superano il rate.
- Se incontrate `RateLimitError`, attendete 60 secondi prima di riprovare.

## Sottosfide

| Subtask | Attacco | Tempo |
|---------|---------|-------|
| A | Direct jailbreak (3 pattern) | 10 min |
| B | Indirect prompt injection (2 email) | 10 min |
| C | Hallucination (3 probe su fatti plausibili-fittizi) | 10 min |
| Final | Compilazione evidence row | 5 min |

## Output attesi

1. `starter.ipynb` con tutti i TODO compilati e log dei tentativi.
2. Tre righe nuove in `shared/evidence_template.csv` (una per subtask).
3. Riflessione finale (1 paragrafo, markdown nel notebook): quale dei tre vettori vi sembra più rischioso *per un sistema bancario reale* e perché?

## ⚠️ Etica

Questi attacchi sono effettuati su un'infrastruttura test (Groq, account vostro, modello open-weights). Non replicate questi pattern su sistemi di produzione altrui senza esplicita autorizzazione scritta — sarebbe accesso non autorizzato a sistema informatico (art. 615-ter c.p.) o, a seconda della giurisdizione, violazione dei termini di servizio del provider.
