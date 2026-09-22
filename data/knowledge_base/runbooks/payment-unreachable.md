# Runbook per `paymentUnreachable`

## Versione e ambito

OpenTelemetry Demo `3.1.0`. Scenario controllato dal feature flag `paymentUnreachable`.

## Effetto atteso

Checkout usa un indirizzo non valido per contattare Payment. Il problema avviene prima che `PaymentService.Charge` possa essere elaborata correttamente.

## Evidenze minime

- valore effettivo di `paymentUnreachable`;
- `trace_id` di un checkout fallito;
- span client Checkout verso Payment;
- log di Checkout nella stessa finestra temporale;
- verifica dell'eventuale assenza dello span server Payment.

## Procedura

1. Aprire la traccia di `CheckoutService.PlaceOrder`.
2. Individuare lo span client diretto a Payment.
3. Esaminare codice di stato, durata ed eventi dello span.
4. Cercare nei log di Checkout l'errore correlato.
5. Verificare se esiste uno span server Payment con lo stesso contesto.
6. Controllare il valore del feature flag.
7. Disattivare il flag nell'ambiente di laboratorio.
8. Ripetere il checkout e verificare il ripristino della chiamata a Payment.

## Diagnosi differenziale

- Uno span server Payment presente e fallito indica che la richiesta ha raggiunto il servizio: considerare `paymentFailure` o un errore applicativo.
- L'assenza di uno span server da sola non prova il flag: verificare anche stato dei container, risoluzione del nome, rete e pipeline di telemetria.
- Se mancano tutti gli span a valle, controllare prima l'esportazione OTLP.

## Risultato da documentare

Riportare valore del flag, errore osservato dal chiamante, presenza o assenza verificata della telemetria Payment ed esito della prova di ripristino.

## Fonti

- [Feature flag della Demo](https://opentelemetry.io/docs/demo/feature-flags/)
- [Servizio Checkout](https://opentelemetry.io/docs/demo/services/checkout/)
- [Servizio Payment](https://opentelemetry.io/docs/demo/services/payment/)
