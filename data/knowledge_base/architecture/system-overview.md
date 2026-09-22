# Architettura del sistema osservato

## Versione di riferimento

Questo documento si riferisce a OpenTelemetry Demo `3.1.0`.

## Scopo della Demo

OpenTelemetry Demo, nota anche come Astronomy Shop, è un'applicazione e-commerce a microservizi costruita per mostrare l'instrumentazione e l'uso congiunto di tracce, metriche e log. I servizi sono sviluppati in linguaggi differenti e comunicano principalmente tramite gRPC. Il browser accede al sistema attraverso il Frontend.

Nel progetto di tesi la Demo è il **sistema sotto osservazione**. L'assistente RAG è un componente esterno: consulta documentazione statica e, negli incrementi successivi, contesto operativo selezionato per aiutare l'analisi degli incidenti.

## Perimetro funzionale iniziale

Il flusso studiato parte dal Frontend e arriva ai servizi coinvolti nella consultazione del catalogo, nella gestione del carrello e nel pagamento.

```text
Browser
  -> Frontend
      -> Product Catalog
      -> Cart -> Valkey
      -> Checkout
          -> Cart
          -> Product Catalog
          -> Payment
          -> altri servizi necessari all'ordine
```

Le frecce indicano dipendenze funzionali, non la direzione con cui vengono esportati i segnali di osservabilità.

## Ruolo dei servizi selezionati

### Frontend

Espone l'interfaccia web e le API usate dal browser. Traduce le azioni dell'utente in chiamate verso i servizi backend. È il punto in cui molti problemi diventano visibili, ma non è necessariamente il punto in cui hanno origine.

### Product Catalog

Fornisce elenco, dettaglio e ricerca dei prodotti. La versione 3.1.0 usa PostgreSQL per la persistenza del catalogo.

### Cart

Gestisce il carrello dell'utente e usa Valkey come archivio. Le operazioni principali sono aggiunta di un articolo, lettura e svuotamento del carrello.

### Checkout

Orchestra il completamento dell'ordine. Recupera il carrello, raccoglie le informazioni sui prodotti, calcola gli importi e invoca i servizi necessari, tra cui Payment. Per questo motivo una richiesta fallita nel Checkout può essere causata da una sua dipendenza.

### Payment

Simula l'addebito e restituisce un identificativo della transazione. Nella Demo non effettua un pagamento reale.

## Dipendenze infrastrutturali rilevanti

- `Valkey` conserva i carrelli.
- `PostgreSQL` conserva i dati del catalogo prodotti.
- `flagd` distribuisce i feature flag usati per attivare scenari controllati.
- OpenTelemetry Collector riceve e instrada la telemetria.
- I backend di osservabilità rendono consultabili tracce, metriche e log.

## Confine tra conoscenza statica e stato runtime

Questi documenti descrivono struttura, responsabilità e procedure. Lo stato reale di un incidente deve invece essere determinato dalle evidenze osservate in una specifica finestra temporale. La knowledge base non deve essere interpretata come prova che un servizio sia sano o guasto.

## Fonti

- [Architettura ufficiale della Demo](https://opentelemetry.io/docs/demo/architecture/)
- [Elenco ufficiale dei servizi](https://opentelemetry.io/docs/demo/services/)
- [Definizione Compose del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/compose.yaml)
- [Protocol Buffer condiviso](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/pb/demo.proto)

