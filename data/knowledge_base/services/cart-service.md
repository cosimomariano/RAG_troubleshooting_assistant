# Servizio Cart

## Versione di riferimento

OpenTelemetry Demo `3.1.0`; servizio `cart`.

## Responsabilità

Cart gestisce i carrelli degli utenti ed è implementato in .NET. Utilizza Valkey come archivio dei dati. Le operazioni gRPC principali sono:

- `AddItem`, per aggiungere un articolo;
- `GetCart`, per leggere il carrello;
- `EmptyCart`, per svuotarlo.

## Dipendenze rilevanti

La dipendenza infrastrutturale principale è `Valkey`. Checkout usa Cart sia per recuperare il contenuto del carrello sia per svuotarlo durante il flusso dell'ordine.

## Telemetria utile

La Demo espone informazioni specifiche, tra cui:

- attributo `demo.cart.items.count`;
- metrica `demo.cart.add_item.latency`;
- metrica `demo.cart.get_cart.latency`.

Queste misure aiutano a distinguere un problema funzionale da un rallentamento dell'accesso al carrello.

## Feature flag associato

Il flag `cartFailure` introduce errori in una percentuale delle chiamate a `EmptyCart`. Lo scenario deve essere trattato come guasto controllato della Demo, non come comportamento casuale dell'infrastruttura.

## Percorso di diagnosi

1. Identificare quale operazione tra `AddItem`, `GetCart` ed `EmptyCart` fallisce.
2. Verificare se l'errore riguarda Cart oppure la comunicazione con Valkey.
3. Controllare il valore corrente di `cartFailure`.
4. Correlare la chiamata con la traccia del Checkout quando il problema appare durante `PlaceOrder`.
5. Confrontare latenza ed errori con una finestra temporale sana.

## Limiti

Un errore durante lo svuotamento del carrello non implica che lettura e aggiunta siano guaste. La diagnosi deve mantenere la granularità dell'operazione.

## Fonti

- [Documentazione ufficiale di Cart](https://opentelemetry.io/docs/demo/services/cart/)
- [Sorgente Cart del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0/src/cart)
- [Configurazione dei feature flag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)

