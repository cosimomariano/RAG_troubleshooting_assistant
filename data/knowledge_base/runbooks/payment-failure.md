# Runbook per `paymentFailure`

## Versione e ambito

OpenTelemetry Demo `3.1.0`. Scenario controllato dal feature flag `paymentFailure`.

## Effetto atteso

Una percentuale delle chiamate a `PaymentService.Charge` fallisce anche se Payment è raggiungibile. Il Checkout può quindi fallire in modo intermittente.

## Evidenze minime

- valore effettivo di `paymentFailure` durante l'incidente;
- `trace_id` di una richiesta fallita;
- span client Checkout verso Payment;
- span server Payment relativo a `Charge`;
- log correlati dei due servizi.

## Procedura

1. Confermare che l'errore si verifichi durante `Charge`.
2. Verificare che nella traccia sia presente lo span server di Payment.
3. Controllare stato, eventi e attributi dello span senza registrare dati sensibili.
4. Confrontare una richiesta fallita con una riuscita nella stessa configurazione.
5. Verificare il valore del flag e la percentuale configurata.
6. Disattivare il flag nell'ambiente di laboratorio.
7. Ripetere più richieste e verificare che gli errori controllati cessino.

## Diagnosi differenziale

- Se manca lo span server Payment e Checkout riporta un errore di connessione, usare il runbook `paymentUnreachable`.
- Se Payment risponde correttamente ma fallisce una fase successiva, continuare il runbook generale del Checkout.
- Se l'errore persiste con flag disattivato, analizzare configurazione, rete e log: il flag non è una spiegazione sufficiente.

## Risultato da documentare

Registrare periodo, valore del flag, tracce rappresentative, operazione coinvolta ed esito della prova con flag disattivato. Non copiare numeri completi di carta.

## Fonti

- [Feature flag della Demo](https://opentelemetry.io/docs/demo/feature-flags/)
- [Servizio Payment](https://opentelemetry.io/docs/demo/services/payment/)
- [Configurazione flagd 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)

