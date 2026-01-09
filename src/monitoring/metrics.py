from prometheus_client import Counter, Histogram, Gauge

# -----------------------
# Request-level metrics
# -----------------------

REQUESTS_TOTAL = Counter(
    "aidr_requests_total",
    "Total extraction requests"
)

REQUEST_FAILURES_TOTAL = Counter(
    "aidr_requests_failed_total",
    "Total failed extraction requests"
)

REQUEST_LATENCY_MS = Histogram(
    "aidr_request_latency_ms",
    "End-to-end extraction latency (ms)",
    buckets=(100, 300, 500, 1000, 2000, 5000, 10000)
)

# -----------------------
# Quality metrics
# -----------------------

HITL_RATE = Gauge(
    "aidr_hitl_rate",
    "Fraction of requests routed to HITL"
)

CONFIDENCE_MEAN = Gauge(
    "aidr_confidence_mean",
    "Mean confidence per request"
)

MISSING_KEYS_TOTAL = Counter(
    "aidr_missing_keys_total",
    "Missing extracted keys",
    ["key"]
)

# -----------------------
# Heuristic observability
# -----------------------

KEY_SYNONYM_USED = Counter(
    "aidr_key_synonym_used",
    "Key synonym matched",
    ["key", "variant"]
)

LAYOUT_STRATEGY_USED = Counter(
    "aidr_layout_strategy_used",
    "Layout strategy used",
    ["strategy"]
)
