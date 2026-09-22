# Contratti gRPC dei servizi selezionati

## Versione di riferimento

OpenTelemetry Demo `3.1.0`. Il contratto autorevole è il file `pb/demo.proto` del tag `3.1.0`.

## Scopo del documento

Questa sintesi collega operazioni, responsabilità e dipendenze per facilitare il retrieval durante il troubleshooting. Non sostituisce il file Protocol Buffer e non deve essere usata per generare automaticamente client o server.

## CartService

### `AddItem`

Riceve l'identificativo dell'utente e un articolo con identificativo prodotto e quantità. Aggiorna il carrello associato all'utente.

### `GetCart`

Riceve l'identificativo dell'utente e restituisce il carrello con gli articoli presenti.

### `EmptyCart`

Riceve l'identificativo dell'utente e rimuove gli articoli dal carrello. Il feature flag `cartFailure` può introdurre errori controllati su questa operazione.

## CheckoutService

### `PlaceOrder`

Riceve i dati necessari al completamento dell'ordine, tra cui utente, indirizzo, valuta e informazioni di pagamento. Restituisce il risultato dell'ordine con i dati prodotti dalle dipendenze, quando il flusso termina correttamente.

È un'operazione orchestrata: un errore restituito da `PlaceOrder` può provenire da Cart, Product Catalog, Currency, Shipping, Payment o Email.

## PaymentService

### `Charge`

Riceve l'importo, la valuta e le informazioni della carta previste dalla simulazione. Restituisce un identificativo della transazione quando l'operazione riesce.

Il feature flag `paymentFailure` agisce sull'esecuzione di `Charge`; `paymentUnreachable` agisce invece sull'indirizzo usato da Checkout per contattare Payment.

## ProductCatalogService

### `ListProducts`

Restituisce l'elenco dei prodotti disponibili.

### `GetProduct`

Riceve l'identificativo del prodotto e restituisce il relativo dettaglio. Il flag `productCatalogFailure` introduce un errore per l'identificativo `OLJCESPC7Z`.

### `SearchProducts`

Riceve una query testuale e restituisce i prodotti compatibili con la ricerca.

## Uso nel troubleshooting

Per ogni errore gRPC occorre registrare:

- servizio chiamante e servizio chiamato;
- nome esatto dell'operazione;
- codice di stato gRPC;
- `trace_id` e span coinvolti;
- eventuali identificatori di dominio non sensibili;
- stato dei feature flag pertinenti.

Non inserire nella knowledge base valori completi di carte, indirizzi personali o altri dati sensibili contenuti nelle richieste.

## Fonte

- [Protocol Buffer della Demo 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/pb/demo.proto)

