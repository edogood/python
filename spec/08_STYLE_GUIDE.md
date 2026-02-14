# Style guide — codice e notebook

## Stile codice (Python)
- Naming: snake_case, funzioni piccole.
- Commenti: spiegare il "perché" per parti non ovvie.
- No scorciatoie opache: evitare lambda complesse, evitare one-liner non leggibili.
- Gestire eccezioni in modo chirurgico:
  - catturare ImportError solo per optional deps
  - catturare Exception solo se si stampa contesto e si decide SKIP
- Non usare seaborn.
- Plot solo se necessario, e solo con matplotlib (dati piccoli).

## Stile notebook
- Ogni sezione scheda:
  - cella markdown: spiegazione completa secondo spec
  - cella(e) code: esempi e assert
- Evitare output enormi: usare head(), sample(), o repr breve.
- Seed fisso per riproducibilità.

## Criteri per "meccanismo" (anti-allucinazione)
Nel testo concettuale:
- descrivere solo ciò che è verificabile o ben noto a livello alto
- vietato inventare dettagli interni specifici (es: "usa hash join X") senza evidenza
- se incerto: dichiarare l'incertezza e limitarsi a descrizione osservabile

## Mini-test
- assert su proprietà stabili (shape, valori attesi su dati costruiti)
- evitare assert su output dipendente da ordinamenti non garantiti, a meno di sorting esplicito
