# Mappa Sfide ↔ AI Act

Riferimento normativo: **Regolamento (UE) 2024/1689** ("AI Act"), in particolare il Capo III ("Sistemi di IA ad alto rischio") e i relativi Allegati III–IV.

## Tabella di mappatura

| Sfida | Subtask | Articolo AI Act | Allegato | Cosa dimostra il controllo |
|-------|---------|-----------------|----------|---------------------------|
| C1 | Misurazione baseline della fairness (4 metriche di gruppo) | **Art. 10** (Data e data governance), **Art. 14** (Sorveglianza umana) | Allegato IV §2 (descrizione del sistema, capacità e limiti) | Esistenza di un protocollo di valutazione del bias *prima* della messa in esercizio |
| C1 | Mitigazione (`ThresholdOptimizer`) e confronto pre/post | **Art. 9** (Sistema di gestione dei rischi) | Allegato IV §2.g (descrizione delle misure di mitigazione) | Tracciabilità delle azioni di mitigazione e misurazione del loro effetto |
| C1 | Documentazione metriche e tradeoff | **Art. 13** (Trasparenza verso utilizzatori a valle) | Allegato IV §3 (informazioni sul monitoraggio) | Comunicazione chiara dei limiti residui post-mitigazione |
| C2 | Robustezza FGSM, sweep ε | **Art. 15** (Accuratezza, robustezza e cibersicurezza) | Allegato IV §3 (test di robustezza) | Esistenza di un test di robustezza adversarial con budget di perturbazione esplicito |
| C2 | Soglia operativa (ε_max @ 80% acc) | **Art. 15** §1, §3 (livello "appropriato" di accuratezza/robustezza) | Allegato IV §2.g | Definizione operazionalizzabile (e dunque verificabile) di "robustezza" |
| C3 | Resistenza a jailbreak diretto | **Art. 15** §5 (cibersicurezza), **Art. 9** | — | Test di sicurezza vs. tentativi di aggirare le policy del sistema |
| C3 | Resistenza a indirect prompt injection | **Art. 15** §5, **Art. 9** | — | Test di sicurezza per il vettore principale di compromissione di LLM integrati con dati esterni |
| C3 | Hallucination | **Art. 13** (Trasparenza), **Art. 9** | Allegato IV §3 | Test della tendenza del modello a fabbricare contenuti, con implicazioni sulla qualità informativa fornita all'utente |

## Note interpretative

### Perché C1 mappa principalmente su Art. 10 e non su Art. 9

L'Art. 10 disciplina la qualità dei dati di addestramento, validazione e test, includendo esplicitamente l'obbligo di **esaminare possibili distorsioni (bias)**. Il fairness audit del laboratorio è quindi parte del *governance plan* dei dati — non un'attività di gestione del rischio in senso stretto. La mitigazione, invece, sì: ricade sotto Art. 9 (sistema di gestione dei rischi, Art. 9.2.d), perché applica una misura tecnica per ridurre un rischio identificato.

In pratica: **il fairness audit appartiene all'Art. 10 quando descrive lo stato del dataset/modello; passa all'Art. 9 quando documenta l'azione correttiva.**

### Perché C2 mappa su Art. 15 e non su Art. 9

L'Art. 15 §1 richiede un livello "appropriato" di accuratezza, robustezza e cibersicurezza. Il §3 specifica che la robustezza deve essere garantita "**anche tramite ridondanza tecnica**". Il §5 entra nel merito della cibersicurezza dichiarando che i sistemi devono essere resistenti a "**tentativi di terzi non autorizzati di alterarne l'uso, gli output o le prestazioni sfruttando vulnerabilità del sistema**".

Gli attacchi avversariali (FGSM, PGD, AutoAttack) sono il caso paradigmatico previsto dal §5. La sfida C2 produce quindi un controllo direttamente riconducibile all'Art. 15.

Il legame con Art. 9 è indiretto: serve a rendere ciclica la valutazione del rischio (Art. 9.2.a — *risk identification* — perché senza un test di robustezza non sai se il rischio esiste), ma il test in sé è un controllo Art. 15.

### Perché C3 mappa su Art. 15 anche per i jailbreak

Il considerando 76 del Regolamento qualifica gli "attacchi tramite *prompt injection*" come una delle minacce di cibersicurezza specifiche dei sistemi di IA. Lo stesso vale per i jailbreak diretti: sono tentativi di alterare le politiche d'uso, e quindi rientrano nella ratio del §5.

L'hallucination è invece più sottile: non è cibersicurezza in senso stretto (non c'è un attaccante che induce il fenomeno), ma riguarda la **qualità informativa** dell'output. Mappa quindi su:

- **Art. 13** (Trasparenza): l'utente a valle deve poter comprendere i limiti del sistema, e l'attitudine alla fabbricazione è un limite materiale;
- **Art. 9** (Risk management): la fabbricazione di citazioni è un rischio operativo significativo (cfr. *Air Canada v. Moffatt*, 2024).

### Sui modelli di IA per finalità generali (GPAI)

Il modello usato in C3 (`llama-3.1-8b-instant`) è un **modello GPAI** ai sensi dell'Art. 3.63. Il sistema costruito in C3 (l'assistente bancario) è invece un sistema di IA ai sensi dell'Art. 3.1, che *integra* un modello GPAI. Le obbligazioni del fornitore del modello GPAI (Art. 53–55) sono distinte da quelle dell'utilizzatore a valle che lo integra in un sistema specifico — sebbene i test di red-team che gli studenti svolgono qui siano analoghi a quelli che il **GPAI provider** deve effettuare per modelli a rischio sistemico (Art. 55.1.b).

Il laboratorio focalizza sull'utilizzatore a valle (operatore di un sistema di credito conversazionale): la responsabilità della sfida C3 è quindi dell'**utilizzatore** che integra il modello, non del fornitore del modello.

## Cosa il laboratorio **non** copre

Per onestà didattica, vale la pena elencare i requisiti dell'AI Act che sono *fuori* dallo scope del laboratorio:

- **Art. 11** (Documentazione tecnica). Le evidence row sono il materiale che alimenta la documentazione tecnica, ma il laboratorio non costruisce un fascicolo Annex IV completo.
- **Art. 12** (Conservazione dei log). I notebook non implementano logging persistente.
- **Art. 14** (Sorveglianza umana). Le sfide simulano la sorveglianza ma non implementano interfacce HITL.
- **Art. 16–22** (Obblighi del fornitore). Lato organizzativo, fuori dallo scope tecnico.
- **Art. 26** (Obblighi del deployer). Idem.

Questi temi sono affrontati nella parte teorica della lezione, non nel laboratorio.
