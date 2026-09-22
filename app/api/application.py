from typing import Protocol, runtime_checkable
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.incident_context import IncidentContextBuilder
from app.api.schemas import ErrorResponse, TroubleshootingRequest, TroubleshootingResponse
from app.generation import LLMClientError
from app.models import RAGResponse

ERROR_RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "La richiesta non può essere elaborata.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "Il body non rispetta il contratto API.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Si è verificato un errore interno.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "Il servizio LLM remoto non è disponibile.",
    },
}


@runtime_checkable
class TroubleshootingService(Protocol):
    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse: ...


class TroubleshootingController:
    def __init__(
        self,
        troubleshooting_service: TroubleshootingService,
        incident_context_builder: IncidentContextBuilder | None = None,
    ) -> None:
        self._troubleshooting_service = troubleshooting_service
        self._incident_context_builder = incident_context_builder or IncidentContextBuilder()

    def troubleshoot(self, request: TroubleshootingRequest) -> TroubleshootingResponse:
        incident_context = self._incident_context_builder.build(request)
        rag_response = self._troubleshooting_service.troubleshoot(
            question=request.question,
            incident_context=incident_context,
        )
        return TroubleshootingResponse.from_rag_response(rag_response)


def create_app(rag_service: TroubleshootingService) -> FastAPI:
    application = FastAPI(
        title="Contratto API dell'assistente RAG per il troubleshooting",
        description=(
            "API per analizzare incidenti tecnici in applicazioni backend "
            "a microservizi tramite una pipeline RAG."
        ),
        version="0.2.0",
    )
    controller = TroubleshootingController(rag_service)

    _register_error_handlers(application)
    _register_routes(application, controller)
    return application


def _register_routes(
    application: FastAPI,
    controller: TroubleshootingController,
) -> None:
    application.post(
        "/troubleshoot",
        response_model=TroubleshootingResponse,
        operation_id="troubleshootIncident",
        summary="Analizza un incidente tecnico tramite il sistema RAG",
        description=(
            "Interroga la Knowledge Base e genera una risposta basata "
            "sulle fonti documentali recuperate."
        ),
        responses=ERROR_RESPONSES,
    )(controller.troubleshoot)


def _register_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        _exception: RequestValidationError,
    ) -> JSONResponse:
        return _create_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="VALIDATION_ERROR",
            message="Il corpo della richiesta non rispetta il contratto API previsto.",
        )

    @application.exception_handler(LLMClientError)
    async def llm_error_handler(
        _request: Request,
        _exception: LLMClientError,
    ) -> JSONResponse:
        return _create_error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="LLM_SERVICE_UNAVAILABLE",
            message="Il servizio LLM remoto non è disponibile.",
        )

    @application.exception_handler(ValueError)
    async def invalid_request_handler(
        _request: Request,
        _exception: ValueError,
    ) -> JSONResponse:
        return _create_error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="La richiesta non può essere elaborata.",
        )

    @application.exception_handler(Exception)
    async def internal_error_handler(
        _request: Request,
        _exception: Exception,
    ) -> JSONResponse:
        return _create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="Errore interno durante l'elaborazione della richiesta.",
        )


def _create_error_response(status_code: int, code: str, message: str) -> JSONResponse:
    error_response = ErrorResponse(code=code, message=message)
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(),
    )