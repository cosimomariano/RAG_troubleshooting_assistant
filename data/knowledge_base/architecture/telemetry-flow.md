# Flusso della telemetria e fonti di osservabilità

## Versione di riferimento

Questo documento si riferisce a OpenTelemetry Demo `3.1.0`.

## Flusso generale

I servizi instrumentati producono log, metriche e tracce. I segnali vengono esportati tramite OTLP verso OpenTelemetry Collector. Il Collector applica la configurazione definita per la Demo e inoltra i dati ai backend di osservabilità.

```text
Servizi della Demo
  -> OTLP
      -> OpenTelemetry Collector
          -> backend delle tracce
          -> backend delle metriche
          -> backend dei log
              -> interfacce di consultazione
```

Il Collector non è un database della knowledge base. È il livello di ricezione, elaborazione ed esportazione della telemetria runtime.

## Tracce

Una traccia rappresenta il percorso di una richiesta tra servizi. Ogni tratto è descritto da uno span. Durante il troubleshooting sono particolarmente utili:

- `trace_id`, per correlare la stessa richiesta tra più servizi;
- `span_id` e relazione padre-figlio, per ricostruire la catena delle chiamate;
- nome del servizio e nome dell'operazione;
- durata e stato dello span;
- attributi ed eventi associati allo span.

La prima operazione fallita nella catena temporale è spesso più informativa dell'errore riportato dal Frontend, ma deve essere verificata insieme ai log.

## Log

I log spiegano eventi applicativi puntuali. Quando sono correlati con `trace_id` e `span_id`, permettono di passare da una traccia anomala al messaggio prodotto dal servizio coinvolto. La sola presenza della parola “error” non è sufficiente a stabilire la causa radice.

## Metriche

Le metriche descrivono l'andamento aggregato del sistema: volumi, latenze, errori e misure specifiche della Demo. Sono utili per individuare il momento di una regressione e confrontare il comportamento corrente con una condizione normale, ma non sostituiscono la singola traccia quando serve seguire una richiesta.

## Correlazione minima per un incidente

Per costruire un contesto operativo utile all'assistente RAG occorrono almeno:

1. intervallo temporale dell'incidente;
2. servizio o funzionalità percepita come guasta;
3. uno o più `trace_id` rappresentativi, quando disponibili;
4. span falliti e relative dipendenze;
5. log correlati;
6. metriche prima, durante e dopo l'evento;
7. stato dei feature flag rilevanti.

## Regole di ingestione nel progetto RAG

La telemetria non deve essere copiata integralmente nella knowledge base statica. Prima dell'indicizzazione occorre:

- selezionare soltanto le evidenze pertinenti;
- normalizzare timestamp e identificatori;
- mascherare dati sensibili;
- conservare la provenienza;
- distinguere chiaramente fatti osservati, ipotesi e procedure generali;
- applicare una politica di conservazione coerente con gli esperimenti.

## Limiti interpretativi

Un attributo dimostrativo può descrivere un comportamento sintetico senza indicare automaticamente un errore tecnico. Per esempio, un pagamento simulato come “non addebitato” non equivale necessariamente a uno span con stato di errore. La diagnosi deve basarsi sull'insieme dei segnali.

## Fonti

- [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)
- [Configurazione del Collector nella Demo 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0/src/otel-collector)
- [Configurazione dei backend di osservabilità](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/compose.observability.yaml)
- [Test ufficiali della telemetria della Demo](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/test/telemetry/README.md)
