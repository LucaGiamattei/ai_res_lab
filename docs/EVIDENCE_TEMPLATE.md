# Il modello dell'**evidence row**

Una *evidence row* è la rappresentazione strutturata di un singolo controllo di test su un sistema di IA. È l'unità atomica su cui si costruiscono i fascicoli di conformità tecnica richiesti dall'AI Act (Art. 11, Allegato IV) e gli evidence pack della piattaforma ai.res.

L'idea pedagogica è semplice: **un test che non produce un artefatto strutturato non è un test riproducibile**. Una metrica stampata in un Jupyter notebook, senza soglia dichiarata né esito formalizzato, non è un controllo di qualità — è un'osservazione.

## Formato

Ogni evidence row contiene almeno le seguenti colonne:

| Colonna | Descrizione | Esempio |
|---------|-------------|---------|
| `challenge` | ID del controllo (qui: C1, C2, C3...) | `C1` |
| `system` | Identificatore del sistema sotto test (modello + dataset + versione) | `GermanCredit-LogReg` |
| `metric` | Metrica adottata, esplicita | `Demographic parity difference` |
| `threshold` | Soglia regolatoria o auto-imposta | `<= 0.10` |
| `observed` | Valore osservato | `0.187` |
| `status` | Esito: `pass` / `fail` / `partial` | `fail` |
| `mitigation` | Mitigazione applicata (o `none`) | `ThresholdOptimizer-EqualizedOdds` |
| `notes` | Osservazioni qualitative, una frase | `Modello favorisce gruppo A; gap oltre soglia regolatoria` |
| `timestamp` | ISO 8601 UTC | `2026-04-28T10:00:00Z` |

## Perché funziona

Tre proprietà non negoziabili:

1. **Atomicità.** Una riga = un controllo. Non si combinano metriche in una riga, e non si rappresenta un controllo su più righe (eccezione: baseline + mitigato sono due righe separate, non due colonne).
2. **Falsificabilità.** Senza `threshold`, il test non è falsificabile: è un'osservazione, non un controllo. La soglia deve essere dichiarata *prima* di osservare il valore (anche se può essere rivista in fase di calibrazione del fascicolo).
3. **Tracciabilità.** Il `timestamp` e la versione del sistema permettono di legare la riga a un commit, a un dataset, a un modello specifico. Senza questi, l'evidence row scade.

## Come tre righe diventano un fascicolo

Le sfide del laboratorio producono almeno cinque evidence row:

```
C1, GermanCredit-LogReg, Demographic parity difference, <= 0.10, ?, ?, none, ...
C1, GermanCredit-LogReg, Demographic parity difference, <= 0.10, ?, ?, ThresholdOptimizer-EqualizedOdds, ...
C2, ImageNet-MobileNetV2, Adversarial robustness (FGSM, L-inf, ε_max @ 80% acc), >= 0.05, ?, ?, none, ...
C3, BankAssistant-Llama3.1-8B, Resilience to direct jailbreak, 0/5, ?/5, ?, none, ...
C3, BankAssistant-Llama3.1-8B, Resilience to indirect prompt injection, 0/2, ?/2, ?, none, ...
C3, BankAssistant-Llama3.1-8B, Resilience to hallucination, 0/3 fabbricazioni, ?/3, ?, none, ...
```

Su `shared/evidence_template.csv`, queste righe formano un *evidence pack* di 6 righe che, accompagnato dai relativi notebook (input/output), dai dataset e dai metadata di versione, costituisce un esempio realistico — minimo ma completo — del materiale da inserire nel fascicolo di conformità tecnica per un sistema ad alto rischio (Annex IV).

## Mappa con la piattaforma ai.res

In ai.res, ogni evidence row è generata da un **runner** (un test eseguibile, versionato, con input/output canonici) e archiviata in un **evidence ledger**. Il ledger:

- mappa ogni riga ad almeno un articolo dell'AI Act (vedi `AI_ACT_MAPPING.md`);
- traccia chi ha eseguito il test, quando, su quale versione del sistema;
- consente di rigenerare il fascicolo Annex IV in formato Word/PDF su richiesta dell'autorità di vigilanza.

Il laboratorio simula manualmente questo flusso. La differenza con un sistema di produzione è che in ai.res i runner sono pianificati (es. fairness audit settimanale), il ledger è immutabile, e l'evidence pack è generato automaticamente — ma l'unità atomica resta sempre la singola evidence row.

## Anti-pattern frequenti

- **Soglia ex post.** Definire la soglia dopo aver visto il valore osservato. Se la soglia dipende dall'osservazione, il controllo non controlla nulla.
- **Status "partial" abusato.** `partial` è ammesso solo se la metrica è multidimensionale (es. fairness su tre gruppi: due passano, uno no). Non è un comodo "non lo so".
- **Mitigation = "none" su test fallito senza commento.** Un fallimento senza piano di rimedio è una passività che non ha posto in un fascicolo di conformità.
- **Notes generiche.** *"Da approfondire"* non è una nota: è un debito tecnico. Le note devono essere specifiche e puntare a un'azione o a un'osservazione utilizzabile.
