from app.api.schemas import (
    LogEvidence,
    MetricEvidence,
    SpanEvidence,
    TroubleshootingRequest,
)


class IncidentContextBuilder:
    def build(self, request: TroubleshootingRequest) -> str | None:
        context_lines: list[str] = []

        self._append_declared_context(context_lines, request.incident_context)
        self._append_service(context_lines, request.service)
        self._append_telemetry(context_lines, request)

        if not context_lines:
            return None
        return "\n".join(context_lines)

    @staticmethod
    def _append_declared_context(context_lines: list[str], incident_context: str | None) -> None:
        if incident_context is None:
            return

        normalized_context = incident_context.strip()
        if normalized_context:
            context_lines.append(f"Contesto dichiarato: {normalized_context}")

    @staticmethod
    def _append_service(context_lines: list[str], service: str | None) -> None:
        if service is None:
            return

        normalized_service = service.strip()
        if normalized_service:
            context_lines.append(f"Servizio principale: {normalized_service}")

    def _append_telemetry(
        self,
        context_lines: list[str],
        request: TroubleshootingRequest,
    ) -> None:
        telemetry = request.telemetry
        if telemetry is None:
            return

        if telemetry.trace_id is not None:
            normalized_trace_id = telemetry.trace_id.strip()
            if normalized_trace_id:
                context_lines.append(f"Trace ID: {normalized_trace_id}")

        for log in telemetry.logs:
            context_lines.append(self._format_log(log))
        for span in telemetry.spans:
            context_lines.append(self._format_span(span))
        for metric in telemetry.metrics:
            context_lines.append(self._format_metric(metric))

    @staticmethod
    def _format_log(log: LogEvidence) -> str:
        details = [f"servizio={log.service}", f"messaggio={log.message}"]
        if log.timestamp is not None:
            details.append(f"timestamp={log.timestamp.isoformat()}")
        if log.severity:
            details.append(f"severità={log.severity}")
        if log.error_type:
            details.append(f"tipo_errore={log.error_type}")
        return f"Log: {', '.join(details)}"

    @staticmethod
    def _format_span(span: SpanEvidence) -> str:
        details = [
            f"servizio={span.service}",
            f"operazione={span.operation}",
            f"stato={span.status}",
        ]
        if span.span_id:
            details.append(f"span_id={span.span_id}")
        if span.peer_service:
            details.append(f"servizio_remoto={span.peer_service}")
        if span.error_message:
            details.append(f"errore={span.error_message}")
        return f"Span: {', '.join(details)}"

    @staticmethod
    def _format_metric(metric: MetricEvidence) -> str:
        details = [f"nome={metric.name}", f"valore={metric.value}"]
        if metric.service:
            details.append(f"servizio={metric.service}")
        if metric.unit:
            details.append(f"unità={metric.unit}")
        return f"Metrica: {', '.join(details)}"


def build_incident_context(request: TroubleshootingRequest) -> str | None:
    return IncidentContextBuilder().build(request)
