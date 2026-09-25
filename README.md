# Progetto di tesi magistrale

## Assistente RAG per il troubleshooting di applicativi basati su microservizi

## Descrizione del problema

Il troubleshooting di applicazioni backend distribuite richiede di correlare informazioni presenti in varie tipologie di documentazione tecnica; testbook, contratti API, log, ticket di segnalazione e incidenti risolti. 
In un'architettura a microservizia, nella quale il componente nel quale un errore diventa visibile può essere diverso da quello in cui il problema ha avuto origine tale operazione assume una difficoltà ancora maggiore.
Questo progetto si pone come scopo quello di sviluppare e valutare sperimentalmente un assistente basato su Retrieval-Augmented Generation (RAG) che supporti l'analisi degli
incidenti, recuperi conoscenza tecnica pertinente dalla knowledge-base e produca risposte verificabili attraverso le fonti utilizzate. Il sistema nella sua totalità è da intendere come uno strumento volto al fine di supporto alle decisioni e non come un agente di remediation autonoma.
Il caso di studio scelto è OpenTelemetry Demo, utilizzato come applicazione a microservizi dalla quale ricavare scenari di guasto ed evidenze di telemetria riproducibili. La versione fissata per il progetto è **OpenTelemetry Demo 3.1.0**.

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

Il confronto sperimentale principale comprende un baseline LLM senza RAG per poi passare alla valutazione di Sparse RAG, Dense RAG, Hybrid RAG e Hybrid RAG con RRF e reranking.

## Knowledge Base

La Knowledge Base iniziale si trova nella cartella `data/knowledge_base` ed è composta da documenti curati per il perimetro Frontend, Checkout, Cart, Payment e Product Catalog di OpenTelemetry Demo 3.1.0.

I documenti sono suddivisi in:

- descrizione dell'architettura e del flusso della telemetria;
- schede dei servizi selezionati;
- sintesi dei contratti gRPC;
- catalogo degli errori controllati tramite feature flag;
- runbook per il troubleshooting;
- regole per la gestione dei dati sensibili.

La Knowledge Base contiene conoscenza tecnica relativamente stabile: log, tracce e metriche prodotte durante l'esecuzione della Demo rappresentano invece evidenze runtime e non vengono indicizzati indiscriminatamente come documenti statici. Il golden dataset sperimentale rimarrà a sua volta separato da entrambi.

La versione di riferimento e le fonti ufficiali utilizzate sono riportate nel documento [`reference-version.md`](data/knowledge_base/architecture/reference-version.md).

## Pipeline del sistema

La pipeline è separata in una fase offline di preparazione della conoscenza e una fase online di analisi delle richieste (inferenza con chiamate REST).

### Pipeline offline

- **Caricamento**: acquisizione controllata di documenti Markdown e testuali.
- **Sensitive Data Detection & Masking**: mascheramento estensibile di PII, credenziali e informazioni infrastrutturali sensibili.
- **Chunking**: creazione di frammenti recuperabili con identificativi stabili e metadati di provenienza. (Section-aware Strategy).
- **Embedding**: trasformazione in batch del testo dei chunk tramite bi-encoder. Implementato.
- **Dense Index**: indicizzazione e persistenza dei vettori e della mappatura dei chunk tramite FAISS. Implementato.
- **Job di indicizzazione**: orchestrazione completa della pipeline offline a partire dalla configurazione. Ancora da implementare.

### Pipeline online

- **Query processing**: composizione della domanda con evidenze di telemetria già normalizzate, senza indicizzare indiscriminatamente i dati runtime grezzi.
- **Dense Retrieval**: embedding della query e ricerca Top-K nell'indice FAISS. Implementato.
- **Sparse Retrieval**: indicizzazione lessicale e ricerca Top-K tramite BM25. Implementato.
- **Hybrid Retrieval**: recupero parallelo sparse e dense con una graduatoria unificata. Implementato.
- **Fusion e reranking**: combinazione tramite RRF e riordinamento opzionale con Cross-Encoder, con misurazione separata delle latenze dei due stadi. Implementato.
- **Prompt assembly**: unione di istruzioni, domanda, contesto dell'incidente, chunk recuperati e riferimenti alle fonti. (eventuale embed dei puntamenti alle fonti come prompt engineering strategy)
- **Generazione**: invocazione REST di un LLM servito da Ollama su una macchina remota della rete privata.
- **Risposta**: restituzione di diagnosi, verifiche suggerite, fonti e metriche di latenza.

## Struttura del progetto

```
|-- app/
|   |-- api/                 # Schemi REST, normalizzazione del contesto e factory FastAPI
|   |-- generation/          # Prompt builder, interfaccia LLM e client Ollama
|   |-- indexing/
|   |   |-- embeddings/      # Astrazione e implementazione Sentence Transformers
|   |   |-- sparse/          # Tokenizzazione tecnica e indice BM25
|   |   |-- vector_store/    # Astrazione e indice persistente FAISS
|   |-- ingestion/
|   |   |-- chunking/        # Chunking section-aware
|   |   |-- loaders/         # Caricamento dei documenti locali
|   |   |-- masking/         # Mascheramento dei dati sensibili
|   |-- models/              # Modelli di dominio
|   |-- retrieval/           # Retriever Dense, Sparse e Hybrid con fusione RRF
|   |-- services/            # Orchestrazione del flusso RAG
|-- configs/
|   |-- application.yaml     # Configurazione applicativa progressiva
|   |-- experiments/         # Configurazioni LLM-only, Dense, Sparse, Hybrid e Hybrid con reranking
|-- contracts/
|   |-- openapi/
|       |-- troubleshooting-api.yaml
|-- data/
|   |-- knowledge_base/      # Corpus documentale OpenTelemetry Demo 3.1.0
|-- tests/                   # Test unitari, contrattuali e di integrazione
|-- .env.example             # Variabili di ambiente di esempio
|-- pyproject.toml           # Dipendenze e configurazione degli strumenti Python
```

## Configurazione

La configurazione di base è descritta in `configs/application.yaml`, mentre i valori dipendenti dall'ambiente sono dichiarati in `.env.example`.

Per creare la configurazione locale su Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Occorre poi valorizzare almeno:

- `OLLAMA_BASE_URL`, con l'indirizzo della macchina che ospita Ollama;
- `OLLAMA_MODEL`, con il nome del modello disponibile sul server remoto;
- `EMBEDDING_MODEL`, con il modello Sentence Transformers scelto;
- `KNOWLEDGE_BASE_PATH`, con il percorso della Knowledge Base;
- `VECTOR_STORE_PATH`, con il percorso in cui salvare l'indice FAISS;
- `RETRIEVAL_MODE`, con una modalità tra `llm_only`, `dense`, `sparse` e `hybrid`;
- `RETRIEVAL_TOP_K`, con il numero massimo di risultati da recuperare;
- `RERANKER_MODEL`, con il modello Cross-Encoder utilizzato per il secondo stadio;
- `RERANKER_BATCH_SIZE`, con il numero di coppie query-chunk valutate per batch;
- `RERANKER_CANDIDATE_TOP_N`, con il numero di candidati RRF inviati al Cross-Encoder.

## Installazione dell'ambiente

Il progetto richiede Python 3.11 o una versione successiva.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Come eseguire il progetto
#TODO COSIMO in corso

## Output del sistema

Il contratto e l'adapter FastAPI definiscono il seguente endpoint REST:

- **POST `/troubleshoot`**: riceve una domanda, un eventuale servizio interessato e un contesto di incidente composto da testo, log, span e metriche normalizzate; restituisce una risposta grounded, le fonti utilizzate e la latenza complessiva.

La specifica completa si trova in [`troubleshooting-api.yaml`](contracts/openapi/troubleshooting-api.yaml).

## Testing

Per eseguire tutti i test:
```powershell
python -m pytest -q
```
La suite attuale comprende **163 casi di test**.

## Note sulla logica di commit e git flow

Ogni commit è composto in questo modo:

```
[RAG]: breve messaggio dell'evolutiva
```

In questo modo è possibile mantenere ordine e tracciabilità delle evolutive nel corso del tempo.
Ogni evolutiva viene sviluppata in un branch dedicato che, successivamente al testing, viene integrato nel ramo `develop` e in seguito rilasciato nel ramo `release`, seguendo la logica di Git Flow.

## Autore

 - Cosimo Mariano