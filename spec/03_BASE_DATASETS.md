# Base datasets (riutilizzabili in tutti i notebook)

## Requisiti
- Seed fisso (riproducibilità).
- Dati piccoli ma realistici: 50–200 righe per tabella.
- Strutture coerenti (chiavi e tipi).

Usare numpy RNG:
- rng = np.random.default_rng(42)

## Tabelle

### 1) sales (vendite)
Colonne:
- date: datetime64[ns] (range giornaliero)
- store_id: int (es. 1..10)
- customer_id: int (es. 1000..1100)
- sku: string/categoria (es. "SKU-001".."SKU-030")
- qty: int (1..5)
- price: float (es. 1.0..100.0)
- discount: float (0..0.30) con molti zeri

Vincoli:
- revenue per riga = qty * price * (1 - discount)

### 2) customers (clienti)
Colonne:
- customer_id: int (chiave; superset di sales.customer_id o subset controllato)
- segment: category/string (es. "consumer","smb","enterprise")
- city: string (es. "Rome","Milan","Turin","Naples","Bologna")
- signup_date: datetime64[ns]

### 3) events (log eventi)
Colonne:
- ts: datetime64[ns] (timestamp a minuti)
- user_id: int (riusa customer_id o range separato)
- event: category/string (es. "view","click","purchase","refund")
- duration_ms: int (0..30000)

## Output del blocco base (obbligatorio)
Ogni notebook deve definire:
- df_sales
- df_customers
- df_events
e anche:
- s_qty (Series derivata, es: df_sales["qty"])
- idx_dates (Index derivato, es: df_sales["date"].sort_values().unique())

## Indicazioni di implementazione
- Implementare il generatore dataset in una singola cella code "BASE_DATASETS".
- Vietato duplicare manualmente dataset in molte celle: una cella unica riusabile per notebook.
- Dopo la creazione:
  - assert su shape (range atteso)
  - assert su dtypes principali
  - esempio: df_sales.head()

## Snippet minimo atteso (linee guida)
- pd.date_range per date
- rng.integers per store_id/customer_id/qty
- rng.uniform per price/discount
- costruzione DataFrame con dtypes stabili
