# Runbook generale per un checkout fallito

## Versione e ambito

OpenTelemetry Demo `3.1.0`. Il runbook riguarda il flusso `Frontend -> Checkout -> dipendenze` e serve a localizzare il primo punto di fallimento.

## Sintomo iniziale

L'utente non riesce a completare l'acquisto oppure il Frontend restituisce un errore durante il checkout. Il sintomo non identifica ancora il servizio responsabile.

## Prerequisiti

- intervallo temporale dell'evento;
- accesso a tracce, log e metriche della Demo;
- stato dei feature flag;
- possibilità di eseguire una singola richiesta di verifica senza alterare dati importanti.

## Procedura

### 1. Delimitare l'incidente

Registrare orario, azione eseguita e comportamento osservato. Stabilire se il problema è costante, intermittente o limitato a uno specifico prodotto o carrello.

### 2. Trovare una traccia rappresentativa

Partire dalla richiesta del Frontend e seguire la traccia fino allo span server di `CheckoutService.PlaceOrder`.

### 3. Identificare il primo errore a valle

Esaminare gli span client di Checkout nell'ordine temporale. Cercare il primo stato di errore o la prima durata anomala verso Cart, Product Catalog, Currency, Shipping, Payment o Email.

### 4. Correlare i log

Usare lo stesso `trace_id` per consultare i log del chiamante e della dipendenza. Separare il messaggio che propaga l'errore dal messaggio che descrive l'origine.

### 5. Controllare le metriche

Confrontare errori e latenza del servizio sospetto prima, durante e dopo l'evento. Verificare se l'anomalia riguarda tutte le operazioni oppure una sola.

### 6. Verificare i feature flag

Controllare almeno `paymentFailure`, `paymentUnreachable`, `cartFailure`, `productCatalogFailure` e `productCatalogLockContention`. Non dedurre il valore del flag dal solo nome dell'errore.

### 7. Formulare la conclusione

La conclusione deve riportare:

- componente e operazione probabilmente responsabili;
- evidenze osservate;
- ipotesi alternative escluse o ancora aperte;
- livello di confidenza;
- azione di verifica o ripristino proposta.

## Criterio di ripristino

Un incidente è considerato rientrato soltanto quando una nuova richiesta completa il flusso e le evidenze mostrano il ritorno a un comportamento normale. Nella Demo, se il guasto è prodotto da un feature flag, disattivare il flag costituisce un ripristino sperimentale, non una correzione del codice.

## Escalation

Se la traccia è assente o incompleta, verificare prima la pipeline OpenTelemetry e il Collector. Non trasformare la mancanza di telemetria in una diagnosi applicativa.

## Fonti

- [Architettura OpenTelemetry Demo](https://opentelemetry.io/docs/demo/architecture/)
- [Servizio Checkout](https://opentelemetry.io/docs/demo/services/checkout/)
- [Feature flag della Demo](https://opentelemetry.io/docs/demo/feature-flags/)
