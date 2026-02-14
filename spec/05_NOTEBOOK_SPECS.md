# Notebook specs — struttura e contenuto

## Regole generali (tutti i notebook)
Ogni notebook deve contenere nell'ordine:

1) Header: titolo e scopo.
2) Ambiente e riproducibilità:
   - stampa/salva: pandas.__version__, sys.version, platform.platform()
   - warning se pandas non è 2.2.*
3) Base datasets (cella unica code): crea df_sales, df_customers, df_events + oggetti derivati.
4) Caricamento inventario:
   - leggere inventory_pandas_api.json
   - filtrare i membri rilevanti per quel notebook (es: target == DataFrame)
5) covered_members:
   - definire set vuoto
   - ogni scheda aggiunge qualname al set
6) Sezioni "scheda" per ogni membro inventariato (1 markdown + 1+ code).
7) Salvataggio coverage:
   - scrivere coverage_<topic>.json con lista ordinata
8) Riepilogo locale:
   - stampare count coperti
   - stampare eventuali membri "skipped" con motivazione

## Mapping notebook -> target
- 01_DataFrame.ipynb -> target "DataFrame"
- 02_Series.ipynb -> target "Series"
- 03_Index.ipynb -> target "Index" (con esempi extra su RangeIndex/DatetimeIndex non inventariati)
- 04_GroupBy.ipynb -> target "GroupBy" (includere esempi sia df.groupby che s.groupby)
- 05_Window.ipynb -> target "Window" (Rolling/Expanding/EWM)
- 06_TopLevel_Functions_and_IO.ipynb -> target "TopLevel" + sezione IO

00_INDEX.ipynb:
- indice dei notebook
- istruzioni per rigenerare con generate_pandas_api_notebooks.py
- descrizione inventory e coverage
- regole di SKIP
- definizione "public API" operativa

99_Coverage_Report.ipynb:
- carica inventory_pandas_api.json
- carica tutti coverage_*.json
- confronta e stampa:
  - mancanti per target
  - alias rilevati
  - deprecati
- se mancanti: descrive come rigenerare (ma il generatore deve anche supportare loop automatico)

## Struttura della scheda (per ogni membro)
Per ogni elemento dell'inventario creare una sezione con:

1) Nome membro (qualname)
2) Categoria: una di:
   - Indexing / Join / Aggregation / Reshape / Cleaning / Window / TimeSeries / IO / Plot / Other
   Regola: categoria determinata da euristica su name + fallback Other (mai inventare).
3) Firma:
   - callable: mostra signature se disponibile, altrimenti spiegare "signature not introspectable"
   - non-callable: mostra type(value) e pattern d'uso
4) Argomenti uno per uno:
   - solo se callable e signature disponibile
   - per ogni parametro:
     - nome
     - default (se presente)
     - effetto
     - edge case tipico
     - errore comune
   Nota: non inferire tipi se non presenti; indicare "type not available".
5) Spiegazione concettuale:
   - problema risolto
   - meccanismo ad alto livello (no dettagli interni non verificati)
   - differenze con membri simili (cross-reference esplicito)
   - quando NON usarlo
6) Esempio minimo (2–10 righe)
7) Esempio realistico (usa dataset base)
8) Edge case + errori comuni (mostrare errore o caso limite)
9) Performance:
   - qualitativa con regole pratiche
   - dichiarare assunzioni (es: "dipende dalla cardinalità" / "materializza copia")
10) Mini-test (assert) stabile

## Accessors / indexers (obbligatori)
Trattare come kind="indexer_accessor" e scheda dedicata:
- DataFrame/Series: loc, iloc, at, iat
- Series: str, dt, cat

Per questi:
- NON usare signature
- mostrare pattern:
  - lettura, scrittura (se applicabile), slicing
- mini-test con assert su risultato

## SKIP policy (obbligatoria)
Se un membro non è esercitabile in modo robusto:
- marcare "SKIPPED" nella cella code
- motivare in markdown
- non aggiungere a covered_members se non viene veramente eseguito
