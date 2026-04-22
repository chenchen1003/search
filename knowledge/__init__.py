import logging

# ChromaDB's posthog telemetry client logs at ERROR when it fails to send
# events (version mismatch: capture() signature changed). Since telemetry is
# already disabled via anonymized_telemetry=False, silence these spurious errors.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
