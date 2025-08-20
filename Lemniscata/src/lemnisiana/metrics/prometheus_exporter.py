from prometheus_client import Counter, Histogram, Gauge

req_latency = Histogram('lem_req_latency_seconds', 'Request latency', buckets=(.05,.1,.25,.5,1,2,5))
cost_tokens = Counter('lem_cost_tokens', 'Total tokens (input+output)', ['service'])
ece_gauge   = Gauge('lem_ece', 'ECE per service', ['service'])
p95_latency = Gauge('lem_latency_p95_ms', 'Latency P95 (ms)', ['service'])
