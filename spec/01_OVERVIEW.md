# Overview — Pandas Public API Notebook Suite

## Obiettivo
Generare una suite di notebook Jupyter + un generatore che coprano in modo completo e verificabile la public API di pandas.
- Target didattico: junior (spiegazioni chiare, esempi minimi e realistici).
- Target pratico: reference da lavoro (firma, edge case, mini-test).

## Ambiente minimo (hard requirement)
- Python 3.11+
- pandas
- numpy

Non assumere dipendenze opzionali per I/O. Ogni esempio che richiede dipendenze extra deve:
- fare import in try/except ImportError
- stampare "SKIPPED: <motivo>"
- NON fallire l'esecuzione del notebook

## Versioning e riproducibilità
All'inizio di ogni notebook:
- stampa e salva in variabili:
  - pandas.__version__
  - sys.version
  - platform.platform()

Assunzione target: pandas==2.2.*.
Se la versione installata non è 2.2.*, continuare ma stampare un warning esplicito:
- "WARNING: pandas version != 2.2.*; inventory may differ."

## Definizione operativa di "public API"
Per ciascun oggetto target, inventariare tutti i membri in:
- dir(obj)
filtrando:
- escludere membri che iniziano con "_"

Per ogni membro, salvare metadati minimi (vedi schema in 04_PUBLIC_API_INVENTORY.md).

## Copertura verificabile
Pipeline obbligatoria:
1) Inventory (JSON) generato da introspezione runtime.
2) Notebook generati che producono "covered_members".
3) Coverage report finale che confronta inventory vs coverage:
   - mancanti
   - alias
   - deprecati
4) Loop di rigenerazione automatico fino a zero mancanti (max 3 iterazioni).

## Output finali
Gli ipynb devono essere JSON validi (nbformat) e tutti i file devono essere generati dal generatore.
