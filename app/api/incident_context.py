from app.api.schemas import (
    LogEvidence,
    MetricEvidence,
    SpanEvidence,
    TroubleshootingRequest,
)


class IncidentContextBuilder:
    def build(self, request: TroubleshootingRequest) -> str | None:
        contextLines: list[str] = []

        self.appendDeclaredContext(contextLines, request.incidentContext)
        self.appendService(contextLines, request.service)
        self.appendTelemetry(contextLines, request)

        if not contextLines:
            return None
        return "\n".join(contextLines)

    @staticmethod
    def appendDeclaredContext(contextLines: list[str], incidentContext: str | None) -> None:
        if incidentContext is None:
            return

        normalizedContext = incidentContext.strip()
        if normalizedContext:
            contextLines.append(f"Contesto dichiarato: {normalizedContext}")

    @staticmethod
    def appendService(contextLines: list[str], service: str | None) -> None:
        if service is None:
            return

        normalizedService = service.strip()
        if normalizedService:
            contextLines.append(f"Servizio principale: {normalizedService}")

    def appendTelemetry(
        self,
        contextLines: list[str],
        request: TroubleshootingRequest,
    ) -> None:
        telemetry = request.telemetry
        if telemetry is None:
            return

        if telemetry.traceId is not None:
            normalizedTraceId = telemetry.traceId.strip()
            if normalizedTraceId:
                contextLines.append(f"Trace ID: {normalizedTraceId}")

        for log in telemetry.logs:
            contextLines.append(self.formatLog(log))
        for span in telemetry.spans:
            contextLines.append(self.formatSpan(span))
        for metric in telemetry.metrics:
            contextLines.append(self.formatMetric(metric))

    @staticmethod
    def formatLog(log: LogEvidence) -> str:
        details = [f"servizio={log.service}", f"messaggio={log.message}"]
        if log.timestamp is not None:
            details.append(f"timestamp={log.timestamp.isoformat()}")
        if log.severity:
            details.append(f"severità={log.severity}")
        if log.errorType:
            details.append(f"tipo_errore={log.errorType}")
        return f"Log: {', '.join(details)}"

    @staticmethod
    def formatSpan(span: SpanEvidence) -> str:
        details = [
            f"servizio={span.service}",
            f"operazione={span.operation}",
            f"stato={span.status}",
        ]
        if span.spanId:
            details.append(f"span_id={span.spanId}")
        if span.peerService:
            details.append(f"servizio_remoto={span.peerService}")
        if span.errorMessage:
            details.append(f"errore={span.errorMessage}")
        return f"Span: {', '.join(details)}"

    @staticmethod
    def formatMetric(metric: MetricEvidence) -> str:
        details = [f"nome={metric.name}", f"valore={metric.value}"]
        if metric.service:
            details.append(f"servizio={metric.service}")
        if metric.unit:
            details.append(f"unità={metric.unit}")
        return f"Metrica: {', '.join(details)}"
