# Runbook per `productCatalogFailure`

## Versione e ambito

OpenTelemetry Demo `3.1.0`. Scenario controllato dal feature flag `productCatalogFailure`.

## Effetto atteso

`ProductCatalogService.GetProduct` fallisce per il prodotto con identificativo `OLJCESPC7Z`. Il guasto è selettivo: altre operazioni o prodotti possono restare disponibili.

## Evidenze minime

- valore effettivo di `productCatalogFailure`;
- `trace_id` della richiesta fallita;
- operazione `GetProduct`;
- attributo `demo.product.id`;
- log correlati di Product Catalog;
- stato di PostgreSQL, se compaiono errori di accesso ai dati.

## Procedura

1. Verificare che il problema riguardi il dettaglio di un prodotto.
2. Controllare che l'operazione sia `GetProduct` e l'identificativo sia `OLJCESPC7Z`.
3. Confrontare il risultato con un prodotto differente.
4. Verificare se `ListProducts` e `SearchProducts` continuano a funzionare.
5. Controllare il valore del feature flag.
6. Disattivare il flag nell'ambiente di laboratorio.
7. Ripetere `GetProduct` per lo stesso identificativo e verificare il ripristino.

## Diagnosi differenziale

- Errori su molti prodotti e operazioni suggeriscono un problema generale del servizio o di PostgreSQL.
- Un errore osservato solo nel Frontend deve essere seguito lungo la traccia prima di attribuirlo a Product Catalog.
- L'assenza dell'attributo prodotto richiede una verifica tramite operazione, log e parametri sanitizzati.

## Risultato da documentare

Registrare identificativo prodotto, stato del flag, tracce rappresentative, confronto con prodotti sani ed esito della prova dopo il ripristino.

## Fonti

- [Feature flag della Demo](https://opentelemetry.io/docs/demo/feature-flags/)
- [Servizio Product Catalog](https://opentelemetry.io/docs/demo/services/product-catalog/)
- [Configurazione flagd 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)
