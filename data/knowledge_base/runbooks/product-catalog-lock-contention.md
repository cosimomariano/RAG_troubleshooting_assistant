# Runbook per `productCatalogLockContention`

## Versione e ambito

OpenTelemetry Demo `3.1.0`. Scenario introdotto dal feature flag `productCatalogLockContention`.

## Effetto atteso

Il flag simula contesa sui lock del database usato da Product Catalog. Il sintomo principale atteso è una degradazione delle operazioni concorrenti, da verificare sulle evidenze runtime; non va presunto un errore per uno specifico prodotto.

## Evidenze minime

- valore effettivo di `productCatalogLockContention`;
- intervallo temporale del rallentamento;
- tracce Product Catalog rappresentative;
- span di accesso a PostgreSQL;
- metriche di latenza ed errore prima e durante lo scenario;
- numero e tipo di richieste concorrenti eseguite nel test.

## Procedura

1. Stabilire una misura di riferimento con il flag disattivato.
2. Attivare lo scenario in un ambiente di laboratorio controllato.
3. Eseguire lo stesso carico e conservare la finestra temporale.
4. Confrontare durata degli span Product Catalog e degli span database.
5. Verificare se la degradazione interessa `ListProducts`, `GetProduct`, `SearchProducts` o più operazioni.
6. Cercare errori o timeout nei log senza considerarli obbligatori: la contesa può manifestarsi principalmente come latenza.
7. Disattivare il flag e ripetere lo stesso carico.
8. Confermare che la latenza ritorni vicina al riferimento.

## Diagnosi differenziale

- `productCatalogFailure` è selettivo per `GetProduct` con identificativo `OLJCESPC7Z`; la contesa riguarda invece il comportamento del database sotto concorrenza.
- Una latenza elevata anche con flag disattivato richiede l'analisi di risorse, rete e configurazione PostgreSQL.
- Un Frontend lento senza span Product Catalog anomali deve essere analizzato lungo un'altra dipendenza.

## Risultato da documentare

Registrare configurazione del carico, finestre confrontate, percentili di latenza, tracce esemplificative, stato del flag ed esito della prova di ripristino. Evitare conclusioni basate su una singola richiesta.

## Fonti

- [Release OpenTelemetry Demo 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/releases/tag/3.1.0)
- [Configurazione flagd 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)
- [Servizio Product Catalog](https://opentelemetry.io/docs/demo/services/product-catalog/)
