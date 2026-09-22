from app.api.schemas import (
    LogEvidence,
    MetricEvidence,
    SpanEvidence,
    TroubleshootingRequest,
)

def build_incident_context(request: TroubleshootingRequest) -> str | None:
    """Raccoglie in un testo stabile le evidenze già normalizzate della richiesta."""
    context_lines: list[str] = []

    if request.incident_context and request.incident_context.strip():
        context_lines.append(f"Contesto dichiarato: {request.incident_context.strip()}")

    if request.service and request.service.strip():
        context_lines.append(f"Servizio principale: {request.service.strip()}")

    telemetry = request.telemetry
    if telemetry is not None:
        if telemetry.trace_id and telemetry.trace_id.strip():
            context_lines.append(f"Trace ID: {telemetry.trace_id.strip()}")

        context_lines.extend(_format_log(log) for log in telemetry.logs)
        context_lines.extend(_format_span(span) for span in telemetry.spans)
        context_lines.extend(_format_metric(metric) for metric in telemetry.metrics)

    if not context_lines:
        return None
    return "\n".join(context_lines)


def _format_log(log: LogEvidence) -> str:
    details = [f"servizio={log.service}", f"messaggio={log.message}"]
    if log.timestamp is not None:
        details.append(f"timestamp={log.timestamp.isoformat()}")
    if log.severity:
        details.append(f"severità={log.severity}")
    if log.error_type:
        details.append(f"tipo_errore={log.error_type}")
    return f"Log: {', '.join(details)}"


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


def _format_metric(metric: MetricEvidence) -> str:
    details = [f"nome={metric.name}", f"valore={metric.value}"]
    if metric.service:
        details.append(f"servizio={metric.service}")
    if metric.unit:
        details.append(f"unità={metric.unit}")
    return f"Metrica: {', '.join(details)}"