## Informazioni circa la configurazione del foundation model utilizzato

Il modello utilizzato come foundation model è fornito da ollama e si tratta di: "ministral-3:8B", dal quale è stato generato un modello custom "rag-thesis-ministral-fixed:v4" con gli iperparametri freezzati.

Il Modelfile nella sua versione 4 presenta i seguenti parametri:
```text
    FROM ministral-3:8b
    PARAMETER temperature 0
    PARAMETER seed 42
    PARAMETER num_predict 3072
    PARAMETER num_ctx 4096
```

### Note sui precedenti Modelfile usati in fase di testing

I precedenti model file presentavano le seguenti configurazioni e relative problematiche:

Modelfile-v1:

    configurazione:
        ```text
            FROM ministral-3:8b
            PARAMETER temperature 0
            PARAMETER seed 42
            PARAMETER num_predict 512
            PARAMETER num_ctx 8192
        ```
    problematica riscontrata: Errore cuda

Modelfile-v2:

    configurazione:
        """text
            FROM ministral-3:8b
            PARAMETER temperature 0
            PARAMETER seed 42
            PARAMETER num_predict 512
            PARAMETER num_ctx 4096
        """

    problematica riscontrata: Le response risultano troncate

Modelfile-v3:

    configurazione:
        ```text
            FROM ministral-3:8b
            PARAMETER temperature 0
            PARAMETER seed 42
            PARAMETER num_predict 1536
            PARAMETER num_ctx 4096
        ```

    problematica riscontrata: 2/10 response sono troncate

## Dati di cattura

Il model file nella sua versione finale è stato creato in data 30/09/2026

## Note

Per la collezione delle metriche della GPU è stato utilizzato il comando:
```powershell
    C.\Windows\System32\nvidia-smi.exe `--query-gpu=utilization.gpu,memory.used,power.draw ` --format=csv,noheader,nounits
```

## Autore

- Mariano Cosimo