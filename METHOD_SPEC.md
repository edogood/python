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

Spiegazione riga per riga (breve ma chiara)

    3–8 righe di spiegazione, solo dove serve.

10) Mini-test (obbligatorio)

    assert che valida il comportamento mostrato.

# assert(s)

11) Correlati (opzionale ma consigliato)

    Link interni ad altre unità API nel notebook:

        “Vedi anche: …”

    Alternative migliori in altri casi (se esistono).

Regole di qualità per le schede

    Nessuna sezione può essere omessa.

    Se una sezione non è determinabile (es. eccezioni), esplicitalo.

    Niente esempi “giganti”: massimo 20–40 righe per esempio minimo.

    Se l’API è altamente parametrica, aggiungi 1 esempio extra (facoltativo) ma sempre breve.


---

## `INVENTORY_AND_COVERAGE.md`

```md
# INVENTORY_AND_COVERAGE.md — Come definire “100%” e misurarlo

## Definizione: public API surface
La copertura al 100% significa: ogni unità API elencata dall’inventario “pubblico” deve avere una scheda.

### Regola base (Python libs)
Include:
- Simboli pubblici in:
  - `package.__all__` (se presente)
  - attributi pubblici del package/module (no underscore) che risultano “user-facing”
- Classi pubbliche e i loro membri pubblici:
  - metodi, classmethod, staticmethod
  - proprietà rilevanti (property) se usate in pratica

Escludi:
- `_private`
- simboli chiaramente interni o di test
- alias puri duplicati (MA: includere alias se fanno parte dell’API e sono comunemente usati; in tal caso, scheda breve che rimanda al canonical)

## Strategie per librerie “giganti”
Per librerie con migliaia di simboli:
- Applica partizione per “surface contract”:
  - Top-level API + classi principali + moduli ufficiali “reference”
- Per SDK generati automaticamente:
  - Copri “client objects” e metodi chiamabili dall’utente
  - Per modelli/DTO generati: documenta pattern e classi base + esempi, e applica un criterio ripetibile (es. tutte le Request/Response principali)
  - Il criterio deve essere scritto nel notebook in “Inventory Summary” e nel coverage report.

## Audit della copertura (obbligatorio)
Il generatore deve produrre un report:
- `inventory_total`
- `documented_total`
- `missing`: lista vuota
- `excluded`: lista con motivazione
- checksum o timestamp inventario

## Come costruire l’inventario (indicazioni)
Metodi tipici (sceglierne uno o combinarli):
1) Introspezione runtime
   - `inspect.getmembers(module)`
   - `dir(obj)` + filtro pubblico
2) `__all__` quando disponibile
3) Package walking
   - `pkgutil.walk_packages` per enumerare submodule (attenzione a import side effects)
4) Per classi “core”: enumerare metodi con `inspect.getmembers(Class, predicate=callable)` + filtro.

## Regole anti-falsi positivi
- Non includere oggetti “import-time heavy” se rompono l’import. In tal caso:
  - segnalarlo e definire un sotto-inventario robusto.
- Se la libreria ha plugin dinamici:
  - inventario baseline + note “plugin-dependent”.

## Definizione: unità API
- Funzione: `module.func`
- Classe: `module.Class`
- Metodo: `module.Class.method`
- Proprietà: `module.Class.prop`
- CLI-first tool: `command subcommand` e `--option` come unità documentabili (vedi LIBRARY_MAP.md)
