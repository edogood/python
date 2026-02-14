# Vincoli non negoziabili

## 1) Eseguibilità senza errori
Tutto deve essere eseguibile senza errori in ambiente minimo:
- Python 3.11+
- pandas, numpy

Per I/O con dipendenze opzionali:
- Parquet: pyarrow o fastparquet (opzionali)
- Excel: openpyxl (opzionale)

Regola: non fallire mai per mancanza dipendenza opzionale.
Implementare:
- try/except ImportError
- output "SKIPPED: missing <package>"

## 2) Niente placeholder
Ogni sezione deve contenere:
- codice eseguibile
- almeno un esempio minimo
- almeno un mini-test assert (se applicabile)

Se un membro non è esercitabile in modo stabile (es: richiede ambiente esterno):
- segnalarlo esplicitamente nella cella markdown
- produrre comunque un "pattern d'uso" + esempio controllato (o skip motivato)
- NON lasciare sezioni vuote

## 3) Copertura verificabile
Ogni notebook deve produrre un set `covered_members` (Python set di stringhe) e salvarlo in JSON:
- coverage_<topic>.json

Formato coverage JSON:
- { "notebook": "...", "covered_members": ["qualname1", ...] }

## 4) Nessuna frase vaga
Ogni descrizione concettuale deve includere SEMPRE:
- problema risolto (cosa ottieni)
- meccanismo (cosa succede ad alto livello)
- quando NON usarlo (anti-pattern chiaro)

## 5) Non assumere I/O esterno
Esempi I/O sempre eseguibili senza file system:
- read_csv / to_csv tramite io.StringIO

Se serve dimostrare path, usare tempfile.TemporaryDirectory e file locali temporanei (consentito).
Non assumere dataset esterni.
