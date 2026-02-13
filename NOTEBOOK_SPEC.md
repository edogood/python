# NOTEBOOK_SPEC.md — Specifica del notebook stile-documentazione

## Struttura minima (ordine obbligatorio)
1) Titolo + scopo + versione libreria (letto a runtime)
2) Indice (TOC) con link/anchor
3) Setup ambiente
   - import
   - seed deterministico (se applicabile)
   - note su versioni
4) “Come leggere questo notebook”
   - definizione di “metodo” / “unità API”
   - convenzioni (signature, param table, mini-test)
5) Sezione: Inventory Summary
   - conteggio totale
   - breakdown per tipo: funzioni / classi / metodi
   - eventuali esclusi con motivazione
6) Sezioni API (core)
   - organizzate per moduli o per classi principali
   - ogni unità API segue METHOD_SPEC.md
7) Appendix
   - Glossario (link a GLOSSARY.md)
   - Troubleshooting
   - Riferimenti interni (cross-link ad altre sezioni)

## Requisiti di navigazione
- Ogni unità API deve avere un anchor stabile.
- Nome visualizzato: `module.symbol` o `Class.method`.
- Per classi molto grandi:
  - sezione classe con overview
  - sottosezioni per metodi

## Requisiti di consultabilità
- Ogni scheda deve poter essere letta isolatamente:
  - cosa fa
  - quando usarla
  - parametri
  - esempio minimo
  - edge case

## Requisiti di esecuzione
- Il notebook deve eseguire end-to-end senza input manuale.
- Sezione esempi: output deterministico dove possibile.
- Eventuali risorse (file) devono essere generate al volo (temp) o incluse in repo.

## Requisiti di consistenza
- Terminologia coerente (usa GLOSSARY.md).
- Parametri spiegati con stessa tabella/ordine.
- Non usare placeholder (TODO, “da completare”, ecc.) nel deliverable finale.
