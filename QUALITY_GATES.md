# QUALITY_GATES.md — Gate di qualità (fail-fast)

Ogni libreria deve passare TUTTI i gate.

## Gate 1 — Esecuzione
- Notebook esegue end-to-end senza errori.
- Nessuna cell richiede input manuale.
- Niente dipendenza di rete non dichiarata.

## Gate 2 — Completezza
- coverage_report: missing = 0
- ogni unità API ha tutte le sezioni METHOD_SPEC.md.

## Gate 3 — Determinismo
- seed fisso dove serve (random, numpy, torch, tensorflow…).
- esempi non basati su timestamp corrente o ordine non deterministico (o dichiarati esplicitamente).

## Gate 4 — Leggibilità reference
- indice presente e funzionante
- anchor stabili
- nomi fully-qualified presenti nelle schede
- parametri in tabella

## Gate 5 — No placeholder / no fluff
- vietati: TODO, “da completare”, “in futuro”, sezioni vuote.
- spiegazioni devono essere verificabili: se ipotesi, dichiararla.

## Gate 6 — Performance “ragionevole”
- esempi leggeri
- evitare training lunghi; per ML usare toy dataset o fit con pochi step.

## Gate 7 — Coerenza terminologica
- termini tecnici definiti al primo uso
- glossario allineato con GLOSSARY.md
