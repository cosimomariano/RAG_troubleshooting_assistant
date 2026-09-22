# Runbook per `cartFailure`

## Versione e ambito

OpenTelemetry Demo `3.1.0`. Scenario controllato dal feature flag `cartFailure` e limitato a `CartService.EmptyCart`.

## Effetto atteso

Una percentuale delle chiamate a `EmptyCart` restituisce un errore. Le operazioni `AddItem` e `GetCart` possono continuare a funzionare.

## Evidenze minime

- valore effettivo di `cartFailure`;
- nome dell'operazione Cart fallita;
- traccia di Checkout e span Cart correlato;
- log del servizio Cart;
- stato di Valkey, se emergono errori di persistenza.

## Procedura

1. Verificare che l'operazione fallita sia `EmptyCart`.
2. Seguire la traccia da `CheckoutService.PlaceOrder` allo span di Cart.
3. Cercare i log Cart con lo stesso `trace_id`.
4. Provare separatamente lettura e aggiunta al carrello per delimitare il guasto.
5. Controllare il valore e la percentuale del feature flag.
6. Disattivare il flag nell'ambiente di laboratorio.
7. Ripetere il checkout e verificare che `EmptyCart` termini correttamente.

## Diagnosi differenziale

- Se falliscono anche `AddItem` e `GetCart`, controllare disponibilità di Cart e Valkey.
- Se `EmptyCart` riesce ma il checkout fallisce dopo, seguire la successiva dipendenza nella traccia.
- Se manca telemetria Cart, verificare l'instrumentazione prima di attribuire la causa.

## Risultato da documentare

Registrare operazione, percentuale configurata, tracce rappresentative e risultato della prova con flag disattivato. Specificare se Valkey è stato escluso come causa.

## Fonti

- [Feature flag della Demo](https://opentelemetry.io/docs/demo/feature-flags/)
- [Servizio Cart](https://opentelemetry.io/docs/demo/services/cart/)
- [Configurazione flagd 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)

