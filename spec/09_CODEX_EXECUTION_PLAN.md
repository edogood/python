# Codex execution plan — istruzioni operative

## Scopo
Usare questi file /spec come unica fonte di verità per generare:
- generate_pandas_api_notebooks.py
- inventory_pandas_api.json
- tutti i .ipynb richiesti

## Regole operative per Codex
1) Leggere /spec/*.md e trattarli come requisiti vincolanti.
2) Implementare prima:
   - build_inventory() e scrittura inventory_pandas_api.json
3) Implementare generazione notebook via nbformat:
   - struttura comune (env + base datasets + inventory load + coverage save)
4) Generare notebook per target in ordine fisso:
   00_INDEX -> 01..06 -> 99
5) Inserire in ogni notebook:
   - salvataggio coverage_<topic>.json
6) Implementare 99_Coverage_Report.ipynb che:
   - legge inventory
   - legge coverage
   - stampa missing/alias/deprecated
7) Garantire determinismo:
   - ordinamenti stabili
   - seed fisso
8) Non introdurre dipendenze oltre:
   - numpy, pandas, nbformat
9) I/O:
   - csv via io.StringIO sempre
   - parquet/excel con SKIP su ImportError

## Definition of done (DoD)
- Lo script genera tutti i file richiesti.
- Tutti i notebook sono JSON validi e apribili.
- Ogni notebook include esempi eseguibili e assert.
- Nessun placeholder.
- SKIP gestito correttamente per dipendenze opzionali.
- Coverage report produce liste mancanti e segnala mismatch.
