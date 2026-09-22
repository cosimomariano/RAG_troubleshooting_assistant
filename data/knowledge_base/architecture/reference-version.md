# Versione di riferimento di OpenTelemetry Demo

## Versione adottata

La knowledge base descrive **OpenTelemetry Demo 3.1.0**, identificata dal tag Git `3.1.0` e dal commit abbreviato `dedc017`. Questa versione costituisce il riferimento riproducibile per il progetto: configurazioni, nomi dei servizi, feature flag e procedure di troubleshooting devono essere verificati rispetto al tag, non rispetto al branch `main`.

La release 3.1.0 è stata pubblicata il 18 settembre 2026. Il repository mantiene licenza Apache 2.0.

## Perché non viene usata la versione 3.0.0

La release 3.0.0 non deve essere utilizzata. Il progetto OpenTelemetry Demo ha segnalato un'incompatibilità di licenza introdotta dal load generator basato su `k6` e `xk6-otel`. La release 3.1.0 ripristina `Locust` come load generator e rimuove i feature flag `loadGeneratorTraffic` e `loadGeneratorVUs`.

La 3.1.0 conserva invece il cambiamento di nomenclatura introdotto nella serie 3.x: gli attributi dimostrativi usano il prefisso `demo.*`, non il precedente `app.*`.

## Perimetro iniziale della knowledge base

Il primo corpus è limitato al flusso di acquisto utile per gli esperimenti di troubleshooting:

- `frontend`;
- `checkout`;
- `cart`;
- `payment`;
- `product-catalog`;
- OpenTelemetry Collector e backend di osservabilità necessari a seguire log, metriche e tracce.

Gli altri servizi della Demo restano dipendenze del sistema, ma non vengono documentati in dettaglio in questo incremento. Saranno aggiunti quando uno scenario sperimentale o un nuovo passo del piano di esecuzione lo richiederà.

## Regole di riproducibilità

1. Clonare o scaricare il tag `3.1.0` della Demo.
2. Registrare ogni eventuale modifica locale alla configurazione Compose.
3. Conservare gli identificatori tecnici originali nei documenti e nei metadati.
4. Considerare la documentazione web ufficiale come guida esplicativa, ma usare il codice del tag `3.1.0` come autorità in caso di differenze.
5. Non mescolare questo corpus statico con log, metriche e tracce raccolti durante gli esperimenti.

## Contenuti esclusi

Questa knowledge base non contiene:

- dump completi di telemetria runtime;
- credenziali, token o dati personali;
- risposte attese del golden dataset;
- incidenti dichiarati risolti senza evidenze riproducibili;
- l'intero repository sorgente della Demo indicizzato senza selezione.

## Fonti

- [Release OpenTelemetry Demo 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/releases/tag/3.1.0)
- [Codice sorgente del tag 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0)
- [Elenco delle release ufficiali](https://github.com/open-telemetry/opentelemetry-demo/releases)
- [Licenza del progetto](https://github.com/open-telemetry/opentelemetry-demo/blob/3.1.0/LICENSE)
