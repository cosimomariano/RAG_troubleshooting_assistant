from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.incident_context import IncidentContextBuilder
from app.api.schemas import ErrorResponse, TroubleshootingRequest, TroubleshootingResponse
from app.generation import LLMClientError, LLMResponseError
from app.services import TroubleshootingSystem

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
    status.HTTP_502_BAD_GATEWAY: {
        "model": ErrorResponse,
        "description": "Il servizio LLM remoto ha restituito una risposta non valida.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "Il servizio LLM remoto non è disponibile.",
    },
}


class TroubleshootingController:
    def __init__(
        self,
        troubleshootingService: TroubleshootingSystem,
        incidentContextBuilder: IncidentContextBuilder | None = None,
    ) -> None:
        self.troubleshootingService = troubleshootingService
        self.incidentContextBuilder = incidentContextBuilder or IncidentContextBuilder()

    def troubleshoot(self, request: TroubleshootingRequest) -> TroubleshootingResponse:
        incidentContext = self.incidentContextBuilder.build(request)
        ragResponse = self.troubleshootingService.troubleshoot(
            question=request.question,
            incidentContext=incidentContext,
        )
        return TroubleshootingResponse.fromRagResponse(ragResponse)


def createApp(ragService: TroubleshootingSystem) -> FastAPI:
    application = FastAPI(
        title="Contratto API dell'assistente RAG per il troubleshooting",
        description=(
            "API per analizzare incidenti tecnici in applicazioni backend "
            "a microservizi tramite una pipeline RAG."
        ),
        version="0.4.0",
    )
    controller = TroubleshootingController(ragService)

    registerErrorHandlers(application)
    registerRoutes(application, controller)
    return application


def registerRoutes(
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


def registerErrorHandlers(application: FastAPI) -> None:
    @application.exception_handler(RequestValidationError)
    async def validationErrorHandler(
        request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        return createErrorResponse(
            statusCode=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="VALIDATION_ERROR",
            message="Il corpo della richiesta non rispetta il contratto API previsto.",
        )

    @application.exception_handler(LLMResponseError)
    async def llmResponseErrorHandler(
        request: Request,
        exception: LLMResponseError,
    ) -> JSONResponse:
        return createErrorResponse(
            statusCode=status.HTTP_502_BAD_GATEWAY,
            code="LLM_INVALID_RESPONSE",
            message="Il servizio LLM remoto ha restituito una risposta non valida.",
        )

    @application.exception_handler(LLMClientError)
    async def llmErrorHandler(
        request: Request,
        exception: LLMClientError,
    ) -> JSONResponse:
        return createErrorResponse(
            statusCode=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="LLM_SERVICE_UNAVAILABLE",
            message="Il servizio LLM remoto non è disponibile.",
        )

    @application.exception_handler(ValueError)
    async def invalidRequestHandler(
        request: Request,
        exception: ValueError,
    ) -> JSONResponse:
        return createErrorResponse(
            statusCode=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message="La richiesta non può essere elaborata.",
        )

    @application.exception_handler(Exception)
    async def internalErrorHandler(
        request: Request,
        exception: Exception,
    ) -> JSONResponse:
        return createErrorResponse(
            statusCode=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="Errore interno durante l'elaborazione della richiesta.",
        )


def createErrorResponse(statusCode: int, code: str, message: str) -> JSONResponse:
    errorResponse = ErrorResponse(code=code, message=message)
    return JSONResponse(
        status_code=statusCode,
        content=errorResponse.model_dump(),
    )
