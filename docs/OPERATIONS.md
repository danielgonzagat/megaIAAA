# Operations Guide

## Setup
- Install dependencies and configure environment variables.
- Provision required services (databases, queues, object storage).
- Run migrations and seed initial data before enabling traffic.

## Feature Flags
- Toggle experimental features using the centralized flag service.
- Flags support gradual rollouts and emergency kills.
- Maintain a changelog for flag additions and removals.

## Monitoring Dashboards
- Use Grafana and Prometheus for system metrics.
- Trace requests with OpenTelemetry and view spans in Jaeger.
- Set up alerts for latency, error rate, and resource saturation.

## Rollback Procedures
- Roll back with `git revert` and redeploy via the CI pipeline.
- Restore database snapshots for stateful services.
- Validate health checks and smoke tests before announcing recovery.

## Shadow and Canary Promotion
```yaml
stages:
  - shadow: 1% traffic
  - canary: 10% traffic
  - full: 100% traffic

promote:
  - guard: p95_latency < 250ms
  - budget: error_rate < 1%
```

## Keeping Docs Synced
- Generate the whitepaper PDF to keep documentation in sync:
  ```bash
  make -C ray/doc latexpdf
  ```
- See [ray/doc/Makefile](../ray/doc/Makefile) for full build instructions.
