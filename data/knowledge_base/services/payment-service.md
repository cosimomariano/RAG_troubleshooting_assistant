# Servizio Payment

## Versione di riferimento

OpenTelemetry Demo `3.1.0`; servizio `payment`.

## Responsabilità

Payment simula l'addebito di un importo su una carta e restituisce un identificativo della transazione. È implementato in JavaScript ed espone l'operazione gRPC `Charge`. Non comunica con un vero circuito di pagamento.

## Dipendenza funzionale

Checkout invoca Payment durante `PlaceOrder`. Di conseguenza, un problema di Payment può apparire all'utente come fallimento generale del checkout.

## Telemetria utile

Gli attributi dimostrativi verificabili includono:

- `demo.payment.amount`;
- `demo.payment.card_type`;
- `demo.payment.card_valid`;
- `demo.payment.charged`;
- `demo.payment.transaction.id`.

Questi valori aiutano a capire se la richiesta è arrivata al servizio e quale risultato ha prodotto la simulazione. Non devono contenere numeri completi di carta o altri dati sensibili.

Nella versione 3.1.0 l'emissione degli attributi grezzi `demo.payment.card_number` e `demo.payment.card_cvv` è protetta dal feature flag `emitRawPii`, disattivato per impostazione predefinita. Non attivarlo negli esperimenti ordinari.

## Feature flag associati

### `paymentFailure`

Introduce errori in una percentuale delle chiamate a `Charge`. In questo caso Payment è raggiungibile, ma l'operazione restituisce un fallimento controllato.

### `paymentUnreachable`

Fa sì che Checkout utilizzi un indirizzo non valido per raggiungere Payment. In questo caso la chiamata non arriva correttamente al servizio Payment: gli span e i log del chiamante diventano quindi centrali.

## Distinzione diagnostica fondamentale

`paymentFailure` e `paymentUnreachable` non sono equivalenti:

- nel primo caso ci si attende evidenza dell'elaborazione nel servizio Payment;
- nel secondo caso ci si attende un errore di connessione o risoluzione osservabile soprattutto dal Checkout, con telemetria Payment assente per la richiesta fallita.

## Limiti

Il valore `demo.payment.charged=false` non deve essere interpretato da solo come prova di un guasto tecnico. Occorre controllare stato dello span, log ed eventuali scenari sintetici attivi.

## Fonti

- [Documentazione ufficiale di Payment](https://opentelemetry.io/docs/demo/services/payment/)
- [Sorgente Payment del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0/src/payment)
- [Configurazione dei feature flag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)
- [Release 3.1.0 e protezione della PII](https://github.com/open-telemetry/opentelemetry-demo/releases/tag/3.1.0)
