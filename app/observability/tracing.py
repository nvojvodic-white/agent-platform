import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource


def setup_tracing():
    resource = Resource.create({
        "service.name": "agent-platform",
        "service.version": "0.1.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development")
    })

    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://jaeger:4317")
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)


def get_tracer():
    return trace.get_tracer("agent-platform")
