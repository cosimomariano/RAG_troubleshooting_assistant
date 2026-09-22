# Servizio Checkout

## Versione di riferimento

OpenTelemetry Demo `3.1.0`; servizio `checkout`.

## Responsabilità

Checkout è il coordinatore del completamento dell'ordine ed è implementato in Go. L'operazione gRPC principale è `PlaceOrder`.

Il servizio recupera i dati del carrello, arricchisce gli articoli con le informazioni del catalogo e coordina i passaggi necessari al pagamento e alla consegna. Una parte delle chiamate viene eseguita verso servizi non ancora inclusi in dettaglio nel corpus, come Currency, Shipping ed Email.

## Dipendenze rilevanti

- `cart`, per leggere e successivamente svuotare il carrello;
- `product-catalog`, per recuperare i dati dei prodotti;
- `currency`, per la conversione degli importi;
- `payment`, per simulare l'addebito;
- `shipping`, per costo e spedizione;
- `email`, per la conferma dell'ordine.

## Telemetria utile

Gli identificatori dimostrativi verificabili nella telemetria ufficiale includono:

- `demo.order.id`;
- `demo.order.amount`;
- `demo.order.items.count`;
- `demo.shipping.amount`;
- `demo.shipping.tracking.id`;
- `demo.payment.transaction.id`.

La presenza di un attributo dipende dal punto raggiunto dall'esecuzione: un ordine interrotto prima del pagamento, per esempio, può non avere un identificativo di transazione.

La versione 3.1.0 mantiene i valori grezzi `user.email`, `demo.payment.card_number` e `demo.payment.card_cvv` dietro il feature flag `emitRawPii`, disattivato per impostazione predefinita. Questi dati non devono essere raccolti nella knowledge base.

## Cosa osservare

1. Lo span server relativo a `PlaceOrder`.
2. L'ordine temporale degli span client verso le dipendenze.
3. Il primo span con stato di errore.
4. Le durate anomale anche in assenza di errore esplicito.
5. I log correlati del Checkout e del servizio a valle coinvolto.

## Interpretazione degli errori

Un errore di Checkout può indicare:

- validazione della richiesta non superata;
- carrello assente o non leggibile;
- prodotto non recuperabile;
- servizio Payment non raggiungibile;
- addebito rifiutato dalla simulazione;
- fallimento di una dipendenza successiva.

Non attribuire automaticamente la causa al Checkout: la sua natura di orchestratore lo rende un punto di propagazione degli errori.

## Fonti

- [Documentazione ufficiale del Checkout](https://opentelemetry.io/docs/demo/services/checkout/)
- [Sorgente Checkout del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0/src/checkout)
- [Contratti gRPC condivisi](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/pb/demo.proto)
- [Release 3.1.0 e protezione della PII](https://github.com/open-telemetry/opentelemetry-demo/releases/tag/3.1.0)
