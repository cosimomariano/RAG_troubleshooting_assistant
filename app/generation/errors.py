"""Modelli associati agli errori prodotti"""

class LLMClientError(RuntimeError): ...
   # Errore generico

class LLMServiceUnavailableError(LLMClientError): ...
    # Servizio non disponibile (es. timeout)

class LLMResponseError(LLMClientError): ...
    # Errore nella validazione della risposta (response non valida)