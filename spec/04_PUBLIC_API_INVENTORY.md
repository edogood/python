# Public API Inventory — Schema e heuristics

## Target objects (inventario via istanze reali)
- DataFrame: df = pd.DataFrame(...)
- Series: s = pd.Series(...)
- Index: idx = pd.Index(...)
  - Mostrare differenze con RangeIndex e DatetimeIndex SOLO come esempi, ma inventario principale resta su pd.Index
- GroupBy:
  - df_gb_type = type(df.groupby("key"))
  - s_gb_type = type(s.groupby(level=0)) (o groupby su chiave)
- Window:
  - Rolling: type(s.rolling(3))
  - Expanding: type(s.expanding())
  - EWM: type(s.ewm(alpha=0.5))
- Top-level:
  - tutti i pubblici di pandas (dir(pd) filtrando "_" e callable True)

## Regola di enumerazione membri
Per ciascun oggetto target:
- members = [m for m in dir(obj) if not m.startswith("_")]

## Record per membro (inventory_pandas_api.json)
Per ogni membro salvare:

- name: str
- kind: "callable" | "property" | "indexer_accessor" | "other"
- qualname: str (es: "pandas.DataFrame.merge")
- deprecated: bool | null
- alias_of: str | null
- signature: str | null

Nota: signature deve essere serializzata come stringa, es: "(self, right, how='inner', ...)".
Se non introspezionabile: null.

## Heuristic: kind
Dato `attr = getattr(obj, name)` (in try/except):
- se name in {"loc","iloc","at","iat"} => kind="indexer_accessor"
- se obj è Series e name in {"str","dt","cat"} => kind="indexer_accessor"
- elif callable(attr) => kind="callable"
- elif isinstance(attr, property) NON è verificabile direttamente (perché getattr restituisce valore); usare euristica:
  - se non callable e non è indexer_accessor e non è (pd.Index, pd.Series, pd.DataFrame, np.ndarray, ...):
    - kind="property" se l'accesso non richiede argomenti e produce scalare o oggetto semplice
  - altrimenti kind="other"
Vincolo: non inventare; se incerto, usare "other".

## Heuristic: signature
Se kind="callable":
- provare inspect.signature(attr)
- se TypeError / ValueError: signature=null
- non crashare mai

## Heuristic: deprecated
Deprecated è "identificabile" solo se:
- docstring contiene pattern robusti:
  - "Deprecated" (case-insensitive)
  - "deprecated" (case-insensitive)
- oppure se chiamare il membro emette DeprecationWarning in modo controllato:
  - usare warnings.catch_warnings(record=True) con simplefilter("always")
  - eseguire chiamata su input minimo sicuro (se possibile)
Se non determinabile senza rischi: deprecated=null (non forzare False).

## Heuristic: alias_of
Stabilire alias_of solo con segnali verificabili:
- se docstring contiene "Alias of" o "alias of" e un riferimento testuale a un altro membro
- oppure se due attributi hanno lo stesso id() (funzioni/metodi) nello stesso oggetto:
  - id(getattr(obj, a)) == id(getattr(obj, b)) => uno è alias dell'altro
Se non determinabile: alias_of=null

## Schema JSON (vincolante)
inventory_pandas_api.json deve essere:
{
  "generated_at_utc": "...",
  "python": "...",
  "platform": "...",
  "pandas_version": "...",
  "target_assumption": "2.2.*",
  "targets": [
    {
      "target": "DataFrame",
      "qualname_root": "pandas.DataFrame",
      "members": [
        {
          "name": "...",
          "kind": "...",
          "qualname": "...",
          "deprecated": true/false/null,
          "alias_of": "..."/null,
          "signature": "..."/null
        }
      ]
    },
    ...
  ]
}

## Ordinamento stabile
Per evitare diff casuali:
- ordinare members alfabeticamente per name
- ordinare targets in ordine fisso:
  DataFrame, Series, Index, GroupBy, Window, TopLevel
