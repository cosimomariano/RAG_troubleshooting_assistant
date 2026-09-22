# Servizio Frontend

## Versione di riferimento

OpenTelemetry Demo `3.1.0`; servizio `frontend`.

## Responsabilità

Il Frontend è l'ingresso principale dell'Astronomy Shop. Espone l'interfaccia web e le API usate dal browser, quindi trasforma le azioni dell'utente in chiamate ai servizi backend. È sviluppato con Next.js.

Nel perimetro iniziale sono rilevanti soprattutto le operazioni che consultano il catalogo, leggono o aggiornano il carrello e avviano il checkout.

## Dipendenze rilevanti

Il Frontend chiama diversi servizi tramite client gRPC. Per il flusso studiato le dipendenze più importanti sono:

- `product-catalog`, per prodotti e ricerca;
- `cart`, per lo stato del carrello;
- `checkout`, per la creazione dell'ordine.

Un errore mostrato dal Frontend può quindi essere soltanto l'effetto di un problema a valle.

## Cosa osservare

### Tracce

- span server della richiesta HTTP ricevuta;
- span client verso il servizio backend;
- codice di stato e durata;
- relazione con il primo span a valle che riporta un errore.

### Log

- messaggi correlati allo stesso `trace_id`;
- nome dell'operazione o della dipendenza chiamata;
- eventuale traduzione dell'errore gRPC in risposta HTTP.

### Metriche

- incremento degli errori HTTP;
- aumento della latenza;
- differenza tra una singola rotta e il comportamento complessivo del Frontend.

## Percorso di diagnosi

1. Individuare la richiesta HTTP che rappresenta il sintomo.
2. Aprire la traccia e seguire i relativi span client.
3. Identificare la prima dipendenza lenta o fallita.
4. Cercare i log del servizio a valle con lo stesso `trace_id`.
5. Verificare se l'errore è isolato a una funzionalità oppure riguarda l'intero Frontend.

## Limiti

Il Frontend non è la causa radice solo perché contiene lo span iniziale o restituisce la risposta visibile all'utente. La responsabilità va attribuita dopo aver esaminato la catena distribuita.

## Fonti

- [Documentazione ufficiale del Frontend](https://opentelemetry.io/docs/demo/services/frontend/)
- [Sorgente Frontend del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0/src/frontend)

