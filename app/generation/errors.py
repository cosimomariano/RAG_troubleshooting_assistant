class LLMClientError(RuntimeError):
    pass


class LLMServiceUnavailableError(LLMClientError):
    pass


class LLMResponseError(LLMClientError):
    pass
