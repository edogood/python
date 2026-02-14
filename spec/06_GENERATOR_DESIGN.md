# Generator design — generate_pandas_api_notebooks.py

## Responsabilità dello script
Lo script deve:
1) Costruire (o ricostruire) inventory_pandas_api.json via introspezione runtime.
2) Generare tutti i notebook .ipynb via nbformat:
   - 00_INDEX.ipynb
   - 01..06
   - 99_Coverage_Report.ipynb
3) Supportare loop di rigenerazione basato su mancanti (max 3 iterazioni):
   - Genera notebook
   - (Opzionale) Esegue notebook? In ambiente minimale non è garantito.
   - Strategia richiesta: generare notebook che salvano coverage_*.json quando l'utente li esegue.
   - Loop automatico nel generatore: consentito SOLO se viene anche eseguito in modo programmatico.
     Se non si esegue, il generatore deve comunque avere la logica per rileggerli e rigenerare.

Nota pratica: Se si implementa esecuzione automatica, usare approccio che non richieda dipendenze extra.
Se non è possibile senza extra, il loop può essere:
- attivabile via flag CLI (es: --execute) e documentato.
- ma la logica di confronto inventory vs coverage deve esistere nello script.

## Vincoli implementativi
- Usare nbformat (nbformat.v4) per costruire notebook JSON validi.
- Ogni "scheda" deve essere:
  - 1 cella markdown
  - 1+ celle code
- Ordinamento deterministic:
  - membri in ordine alfabetico
  - notebook sempre generati nello stesso ordine

## Moduli standard consentiti
- json, os, sys, platform, inspect, warnings, textwrap, io, tempfile, datetime
- numpy, pandas
- nbformat

## Architettura suggerita (funzioni)
- get_env_metadata() -> dict
- make_base_datasets_code_cell() -> nbformat cell
- build_inventory() -> dict (schema come 04_PUBLIC_API_INVENTORY.md)
- write_inventory_json(inventory, path)
- categorize_member(name, target) -> str (euristica deterministica)
- make_member_cells(target, member_record) -> list[cells]
- build_notebook_for_target(target_name, member_records, topic_id) -> nb
- build_index_notebook(...) -> nb
- build_coverage_report_notebook(...) -> nb
- write_notebook(nb, path)

## Euristiche categoria (deterministiche)
Implementare mapping basato sul nome (lowercase) e target:

Esempi (non esaustivo; deve essere estendibile):
- Indexing: {"loc","iloc","at","iat","xs","__getitem__" (se presente come name), "take"}
- Join: {"merge","join","concat"} (top-level concat)
- Aggregation: {"groupby","agg","aggregate","sum","mean","min","max","count","value_counts","nunique"}
- Reshape: {"pivot","pivot_table","melt","stack","unstack","wide_to_long","explode","crosstab","get_dummies"}
- Cleaning: {"drop","dropna","fillna","replace","astype","rename","clip","where","mask"}
- Window: {"rolling","expanding","ewm"}
- TimeSeries: {"resample","to_datetime","date_range","shift","tz_localize","tz_convert"}
- IO: {"read_","to_"} + notebook IO section (StringIO + try/except)
- Plot: {"plot","hist","boxplot"} (ma usare matplotlib solo se serve)
Fallback: Other

Regola: se non matcha nulla, "Other".

## Gestione membri problematici
Molti metodi pandas richiedono contesti o dipendenze non minime.
Regola: per ogni membro, provare:
- esempio minimo controllato
Se l'esempio fallisce per ragioni strutturali non gestibili (es: richiede motore esterno):
- SKIPPED con motivazione
- Non aggiungere a covered_members

## Contenuto code per scheda
Ogni scheda deve avere un blocco code con questa struttura logica:

- recupero dell'oggetto base (df/s/idx/gb/window)
- accesso membro: getattr(obj, name) o pattern accessor
- esempio minimo
- esempio realistico
- edge case
- mini-test assert
- se passa: covered_members.add(qualname)

In caso di ImportError su optional IO:
- stampare SKIPPED e non fallire
