# EXAMPLES_AND_TESTS.md — Regole per esempi e mini-test

## Obiettivi
- Esempi piccoli ma significativi.
- Mini-test che validano l’esempio, non l’intera libreria.

## Regole generali
1) Preferisci input sintetici piccoli (liste, array 3x3, DataFrame con 5 righe).
2) Evita output troppo lunghi: stampa solo head / shape / dtype / summary.
3) Ogni esempio deve avere almeno 1 assert.
4) Ogni esempio deve spiegare almeno 1 pitfall o edge case collegato.

## Seed policy
- Usa un seed unico “notebook-wide” e documentalo nella sezione Setup.
- Per librerie diverse:
  - numpy/random
  - torch manual seed
  - tensorflow set_seed
  - python random.seed

## Test policy
- Mini-test nel notebook: assert semplici.
- Se il progetto include test suite esterna (opzionale):
  - estrai esempi in funzioni e testa con pytest.

## Pattern consigliati per ML
- Dataset toy (es. make_classification) o sintetico.
- Fit rapidissimo (pochi estimators / pochi epochs / pochi steps).
- Assert su shape e su proprietà basilari (non su metriche instabili).
