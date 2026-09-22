# Gestione dei dati sensibili nella telemetria

## Versione di riferimento

OpenTelemetry Demo `3.1.0` e pipeline di ingestione del progetto RAG.

## Protezioni disponibili nella Demo

La release 3.1.0 protegge l'emissione di attributi grezzi potenzialmente sensibili tramite il feature flag `emitRawPii`, disattivato per impostazione predefinita. Gli attributi interessati sono:

- `user.email`;
- `demo.payment.card_number`;
- `demo.payment.card_cvv`.

La configurazione del Collector contiene inoltre esempi di eliminazione, hashing e mascheramento parziale tramite transform processor, affiancati dal redaction processor come protezione basata sui nomi delle chiavi.

## Regola per gli esperimenti

`emitRawPii` deve rimanere disattivato. La sua presenza è utile per studiare le misure di sicurezza della Demo, non per raccogliere dati grezzi. Nessun numero completo di carta, codice CVV, indirizzo email personale, token o credenziale deve entrare nella knowledge base o nel golden dataset.

## Difesa a più livelli

La sicurezza non deve dipendere da un solo controllo:

1. evitare l'emissione alla fonte;
2. redigere o eliminare gli attributi nel Collector;
3. selezionare soltanto la telemetria necessaria;
4. applicare il masker dell'ingestion pipeline prima del chunking;
5. verificare i documenti trasformati prima dell'indicizzazione;
6. conservare provenienza e politica di retention.

## Dati ammessi nella knowledge base

Sono ammessi identificatori tecnici utili alla diagnosi quando non rappresentano segreti o dati personali, per esempio:

- nome del servizio e dell'operazione;
- codice di stato;
- durata;
- `trace_id` e `span_id` pseudonimi relativi a esperimenti controllati;
- identificativi dei prodotti sintetici della Demo;
- nome e valore non sensibile di un feature flag.

## Dati da mascherare o escludere

- dati della carta e CVV;
- email e indirizzi personali;
- password, token, chiavi e stringhe di connessione;
- header di autenticazione;
- payload completi non necessari alla diagnosi;
- informazioni provenienti da ambienti esterni alla Demo senza autorizzazione.

## Verifica prima dell'indicizzazione

Un documento derivato dalla telemetria può essere indicizzato solo se:

- la fonte e la finestra temporale sono note;
- i campi sensibili sono stati rimossi o mascherati;
- il contenuto è pertinente allo scenario;
- fatti osservati e interpretazioni sono separati;
- non contiene una risposta attesa del dataset sperimentale.

## Fonti

- [Release OpenTelemetry Demo 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/releases/tag/3.1.0)
- [Configurazione del Collector 3.1.0](https://github.com/open-telemetry/opentelemetry-demo/tree/3.1.0/src/otel-collector)
- [Gestione dei dati sensibili in OpenTelemetry](https://opentelemetry.io/docs/security/handling-sensitive-data/)
