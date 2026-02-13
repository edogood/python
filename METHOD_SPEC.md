# METHOD_SPEC.md — Template “scheda unità API” (funzione/classe/metodo)

Ogni unità API deve essere documentata con questa struttura.

## 1) Identità
- **Nome**: `<fully_qualified_name>`
- **Tipo**: function | class | method | property | cli-command | config-option
- **Signature**: riportare esattamente la signature (da inspect o equivalente)
- **Disponibilità**: (opzionale) since / deprecated / experimental se rilevabile

## 2) Spiegazione in linguaggio semplice (1–3 frasi)
- Cosa fa, senza jargon.
- Se il concetto è avanzato, definisci i termini tecnici in 1 frase ciascuno.

## 3) Analogia concreta (1 breve)
- Un parallelo reale che chiarisca l’uso.

## 4) Meccanismo (causa → effetto)
- Spiegare cosa succede “sotto”: input → trasformazione → output.
- Se rilevante: complessità, mutabilità, allocazioni, lazy vs eager, side effects.

## 5) Parametri (tabella obbligatoria)
| Parametro | Tipo | Default | Significato | Vincoli / Edge case |
|---|---|---:|---|---|
| ... | ... | ... | ... | ... |

Note:
- Se `*args/**kwargs`: esplicitare i pattern d’uso e i casi comuni.
- Se parametri accettano più tipi: documentare branch principali.

## 6) Return / Output (obbligatorio)
- Tipo e significato.
- Se ritorna generator/lazy object: spiega quando avviene il calcolo.
- Se muta oggetti: chiarire cosa cambia e cosa no.

## 7) Errori ed eccezioni (obbligatorio)
- Elenco delle eccezioni comuni e quando si verificano.
- Se non chiaro: indicare “Non garantito; verificare docstring”.

## 8) Pitfall e edge case (obbligatorio)
- Almeno 2–5 bullet:
  - casi limite
  - performance trap
  - comportamenti sorprendenti
  - incompatibilità frequenti

## 9) Esempio minimo (obbligatorio)
- Un esempio “small and sharp”.
- Deve essere eseguibile e deterministico.

### Codice
```python
# esempio
