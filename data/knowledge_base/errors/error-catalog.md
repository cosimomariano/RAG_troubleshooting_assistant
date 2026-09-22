# Catalogo degli errori controllati

## Versione di riferimento

OpenTelemetry Demo `3.1.0`; configurazione `src/flagd/demo.flagd.json`.

## Scopo

La Demo usa feature flag per introdurre malfunzionamenti riproducibili. Questo catalogo descrive gli scenari inclusi nel perimetro iniziale senza inventare messaggi di errore: testo esatto, codici e timestamp devono essere acquisiti dall'esecuzione osservata.

## `paymentFailure`

### Effetto previsto

Una percentuale delle chiamate a `PaymentService.Charge` restituisce un errore controllato.

### Evidenze da cercare

- span client di Checkout verso Payment;
- span server e log di Payment per la stessa traccia;
- stato di errore dell'operazione `Charge`;
- aumento degli errori durante il checkout.

### Diagnosi differenziale

Payment è raggiungibile. Se manca completamente lo span server di Payment, considerare invece un problema di connettività o lo scenario `paymentUnreachable`.

## `paymentUnreachable`

### Effetto previsto

Checkout tenta di contattare Payment tramite un indirizzo non valido.

### Evidenze da cercare

- errore nello span client prodotto da Checkout;
- log di Checkout relativi alla chiamata;
- assenza dello span server Payment per la richiesta fallita;
- eventuali errori di connessione o risoluzione riportati a runtime.

### Diagnosi differenziale

Il problema è nel raggiungimento della dipendenza, non nell'elaborazione interna di `Charge`.

## `cartFailure`

### Effetto previsto

Una percentuale delle chiamate a `CartService.EmptyCart` fallisce.

### Evidenze da cercare

- operazione esatta `EmptyCart`;
- span e log del servizio Cart;
- traccia del `PlaceOrder` che ha invocato lo svuotamento;
- normale funzionamento eventuale di `AddItem` e `GetCart`.

### Diagnosi differenziale

Non confondere il fallimento selettivo di `EmptyCart` con l'indisponibilità di Valkey o dell'intero servizio Cart.

## `productCatalogFailure`

### Effetto previsto

`ProductCatalogService.GetProduct` fallisce per il prodotto `OLJCESPC7Z`.

### Evidenze da cercare

- attributo `demo.product.id` con valore `OLJCESPC7Z`;
- span e log di `GetProduct`;
- eventuale propagazione verso Frontend o Checkout;
- esito positivo delle richieste per prodotti differenti.

### Diagnosi differenziale

Lo scenario è selettivo e non dimostra un guasto generale di PostgreSQL o di Product Catalog.

## Flag non presenti nella versione 3.1.0

I flag `loadGeneratorTraffic` e `loadGeneratorVUs`, introdotti nella release 3.0.0, sono stati rimossi nella 3.1.0 insieme al ritorno a Locust. Non devono comparire nelle procedure operative di questa knowledge base.

## `productCatalogLockContention`

### Effetto previsto

La release 3.1.0 aggiunge un comportamento controllato che simula contesa sui lock del database usato da Product Catalog.

### Evidenze da cercare

- aumento della latenza delle operazioni Product Catalog;
- span di accesso a PostgreSQL più lunghi del comportamento di riferimento;
- richieste concorrenti interessate nello stesso intervallo;
- metriche di durata ed errore del servizio;
- valore del flag durante l'esperimento.

### Diagnosi differenziale

La contesa non equivale al fallimento mirato di `GetProduct` prodotto da `productCatalogFailure`. Confrontare più operazioni e più identificativi di prodotto.

## Regola di validazione

Prima di concludere che un feature flag sia la causa radice, verificare il suo valore effettivo nella sessione sperimentale e raccogliere almeno una traccia coerente con l'effetto documentato.

## Fonti

- [Feature flag della Demo](https://opentelemetry.io/docs/demo/feature-flags/)
- [Configurazione flagd del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/src/flagd/demo.flagd.json)
- [Release 3.1.0 e rimozione dei flag del load generator](https://github.com/open-telemetry/opentelemetry-demo/releases/tag/3.1.0)
