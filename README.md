# Progetto di tesi magistrale

## Assistente RAG per il troubleshooting di applicativi basati su microservizi

## Descrizione del problema

Il troubleshooting di applicazioni backend distribuite richiede di correlare informazioni presenti in varie tipologie di documentazione tecnica; testbook, contratti API, log, ticket di segnalazione e incidenti risolti. 
In un'architettura a microservizia, nella quale il componente nel quale un errore diventa visibile può essere diverso da quello in cui il problema ha avuto origine tale operazione assume una difficoltà ancora maggiore.
Questo progetto si pone come scopo quello di sviluppare e valutare sperimentalmente un assistente basato su Retrieval-Augmented Generation (RAG) che supporti l'analisi degli
incidenti, recuperi conoscenza tecnica pertinente dalla knowledge-base e produca risposte verificabili attraverso le fonti utilizzate. Il sistema nella sua totalità è da intendere come uno strumento volto al fine di supporto alle decisioni e non come un agente di remediation autonoma.
Il caso di studio previsto è OpenTelemetry Demo, il quale sarà utilizzato come applicazione a microservizi dalla quale ricavare scenari di guasto ed evidenze di telemetria riproducibili.

## Obiettivi

L'obiettivo principale è confrontare in modo controllato diverse configurazioni di retrieval e misurarne l'efficacia nel troubleshooting tecnico. 
Il progetto prevede:

1. ingestione, normalizzazione e mascheramento dei dati sensibili presenti nella Knowledge-Base;
2. suddivisione deterministica dei documenti in chunk con metadati (batching) e provenienza;
3. retrieval sparso tramite BM25 e retrieval denso tramite bi-encoder;
4. retrieval ibrido con Reciprocal Rank Fusion (RRF);
5. riordinamento dei candidati tramite Cross-Encoder;
6. costruzione di prompt tramite prompt engineering (sfruttando i puntamenti alle fonti del RAG) e generazione tramite un modello servito da Ollama su una macchina remota configurabile;
7. valutazione di qualità, faithfulness, retrieval e prestazioni operative.

Il confronto sperimentale principale comprende un baseline LLM senza RAG per poi passare alla valutazione di: sparse RAG, dense RAG, hybrid RAG e hybrid RAG con RRF e reranking.

## Pipeline del sistema

La pipeline è separata in una fase offline di preparazione della conoscenza e una fase online di analisi delle richieste (inferenza con chiamate REST).

### Pipeline offline

- **Caricamento**: acquisizione controllata di documenti Markdown e testuali.
- **Sensitive Data Detection & Masking**: mascheramento estensibile di PII, credenziali e informazioni infrastrutturali sensibili.
- **Chunking**: creazione di frammenti recuperabili con identificativi stabili e metadati di provenienza.
- **Embedding e indicizzazione**: generazione di rappresentazioni dense e costruzione degli indici necessari al retrieval.

### Pipeline online

- **Query processing**: composizione della domanda con evidenze di telemetria normalizzate, senza indicizzare indiscriminatamente i dati runtime grezzi.
- **Retrieval**: selezione configurabile tra modalità sparsa, densa e ibrida.
- **Fusion e reranking**: combinazione delle graduatorie con RRF e riordinamento opzionale tramite Cross-Encoder.
- **Prompt assembly**: unione di istruzioni, domanda, contesto dell'incidente, chunk recuperati e riferimenti alle fonti. (eventuale embed dei puntamenti alle fonti come prompt engineering strategy)
- **Generazione**: invocazione REST di un LLM servito da Ollama su una macchina remota della rete privata.
- **Risposta**: restituzione di diagnosi, verifiche suggerite, fonti e metriche di latenza.

## Struttura del progetto

 # TODO COSIMO IN CORSO DI DEFINIZIONE
```
|-- contracts/
|   |-- openapi/
|       |-- troubleshooting-api.yaml
```

## Come eseguire il progetto

 # TODO COSIMO 

## Output del sistema 

Il sistema esporrà un endpoint REST principale:

- **POST `/troubleshoot`**: riceve domanda e contesto dell'incidente e restituisce una risposta grounded, le fonti utilizzate e la latenza complessiva.


## Testing

# TODO COSIMO

## Note sul codice

# TODO COSIMO

## Autore

 - Cosimo Mariano