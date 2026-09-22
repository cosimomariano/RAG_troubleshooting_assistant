# Servizio Product Catalog

## Versione di riferimento

OpenTelemetry Demo `3.1.0`; servizio `product-catalog`.

## Responsabilità

Product Catalog gestisce la consultazione dei prodotti ed è implementato in Go. Nella versione di riferimento usa PostgreSQL. Le operazioni gRPC principali sono:

- `ListProducts`, per ottenere l'elenco;
- `GetProduct`, per ottenere un prodotto tramite identificativo;
- `SearchProducts`, per eseguire una ricerca testuale.

## Dipendenze e chiamanti

La dipendenza dati principale è PostgreSQL. Frontend consulta direttamente il catalogo; Checkout usa Product Catalog per arricchire gli articoli presenti nel carrello.

## Telemetria utile

Gli attributi dimostrativi verificabili includono:

- `demo.product.id`;
- `demo.product.name`;
- `demo.product.count`;
- `demo.product.search.count`.

Questi attributi permettono di collegare un errore alla specifica operazione o al prodotto coinvolto senza affidarsi soltanto al messaggio mostrato nel Frontend.

## Feature flag associati

Il flag `productCatalogFailure` fa fallire `GetProduct` per il prodotto con identificativo `OLJCESPC7Z`. Lo scenario è selettivo: l'elenco e gli altri prodotti possono continuare a funzionare.

La release 3.1.0 aggiunge inoltre `productCatalogLockContention`, che simula contesa sui lock del database. Questo scenario va analizzato principalmente come degradazione di latenza e concorrenza, verificando gli span di accesso a PostgreSQL; non deve essere confuso con il fallimento selettivo di uno specifico prodotto.

## Percorso di diagnosi

1. Identificare l'operazione interessata.
2. Se si tratta di `GetProduct`, controllare `demo.product.id`.
3. Verificare lo stato di `productCatalogFailure` e `productCatalogLockContention`.
4. Distinguere un errore applicativo da un problema di accesso a PostgreSQL.
5. Correlare la richiesta con gli span del Frontend o del Checkout.

## Limiti

Un singolo prodotto non recuperabile non dimostra che l'intero catalogo sia indisponibile. Verificare separatamente elenco, ricerca e dettaglio.

## Fonti

- [Documentazione ufficiale di Product Catalog](https://opentelemetry.io/docs/demo/services/product-catalog/)
- [Sorgente Product Catalog del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0/src/product-catalog)
- [Configurazione dei feature flag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)
- [Release 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/releases/tag/3.1.0)
