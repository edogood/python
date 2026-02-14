# File tree e artefatti target

## Artefatti obbligatori (root del repo)
- generate_pandas_api_notebooks.py
- inventory_pandas_api.json
- 00_INDEX.ipynb
- 01_DataFrame.ipynb
- 02_Series.ipynb
- 03_Index.ipynb
- 04_GroupBy.ipynb
- 05_Window.ipynb
- 06_TopLevel_Functions_and_IO.ipynb
- 99_Coverage_Report.ipynb

## Artefatti intermedi (root del repo)
- coverage_dataframe.json
- coverage_series.json
- coverage_index.json
- coverage_groupby.json
- coverage_window.json
- coverage_toplevel.json

Nota: questi file coverage_*.json sono prodotti dai notebook (o dal generatore) e consumati da 99_Coverage_Report.ipynb.

## Cartella spec (solo documentazione)
- /spec/*.md (questi file)
