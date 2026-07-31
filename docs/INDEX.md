# Documentation Index

## Start here

The current architecture and decision-authority rules are defined by
[Distributed Execution: Source of Truth](../DISTRIBUTED_EXECUTION_SOURCE_OF_TRUTH.md).
It is the authority whenever another document disagrees with it.

The supported workflow is simple:

1. Configure the registered local policies through a `CanonicalVector`.
2. Select that vector offline with the GA and graph-family evaluation tools.
3. Evaluate every candidate and run every experiment through the distributed
   runtime.
4. Let nodes communicate and finish locally as matched or terminal-unmatched.

## Current references

- [ALGORITHM_FLOW.md](ALGORITHM_FLOW.md) — high-level runtime flow
- [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md) — vector schema and configuration
- [GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md) — offline vector search
- [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) — graph families and robust scoring
- [GRAPH.md](GRAPH.md) — graph topology API
- [METRICS.md](METRICS.md) — measurements and reports
- [NOTEBOOKS.md](NOTEBOOKS.md) — notebook workflow
- [DEVELOPMENT.md](DEVELOPMENT.md) — local development checks

## Historical material

Older documents may describe removed centralized execution, state stores, or
queue-based runtime machinery. They are retained only as historical design
records and are not an implementation guide. Do not use them to infer the
current runtime API.

The dated audits, specifications, and implementation plans under
`docs/superpowers/` are likewise historical records; their status should be
read in the context of their date.
