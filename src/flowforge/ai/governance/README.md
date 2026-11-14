"""Architecture and Technical Contract: AI Budget Ledger & Governance Core.

# Architecture Document: AI Budget/Audit Policy Core (M0-018B1)

## 1. Phased Boundary and Non-Claims
This module (`src/flowforge/ai/governance`) provides a pure, persistence-neutral policy and repository contract.
- It defines versionable, bounded, explicit tenant/run/provider-scoped budget settings, immutable reservation / attempt / usage / settlement records, and an abstract repository Protocol (`BudgetRepository`).
- A deterministic in-memory reference repository (`InMemoryBudgetRepository`) is provided for unit and concurrency testing.
- **IMPORTANT**: This phase (M0-018B1) DOES NOT claim working durable DB-backed limiting, Celery task wiring, or production PostgreSQL migrations. Those are specifically scheduled for M0-018B2 following integration reviews.

## 2. Hard External Billing Guarantees vs Token Estimation
- Hard external billing guarantees are mathematically and architecturally impossible from token estimation alone. Upstream LLM providers bill on tokens generated on their internal infrastructure; network disconnections, streaming interruptions, or abrupt client cancellations may leave tokens billed upstream without delivery to the client.
- To prevent financial leakage, FlowForge governance explicitly distinguishes:
  1. `ZERO_COST_REJECTED`: Call rejected locally prior to network dispatch (e.g. rate limit, budget exhaustion, validation failure). Counted as 0 cost, 0 tokens.
  2. `ESTIMATED_EXPOSURE`: Call dispatched to provider, but connection timed out or response dropped. Retains conservative estimated exposure to reflect potential provider debit.
  3. `EXACT`: Verified token tallies and actual cost returned by provider.
  4. `UNKNOWN_USAGE`: Actual usage could not be determined; preserves estimated exposure conservatively.

## 3. Idempotency and Deduplication Contract
- Every reservation requires a caller-supplied or generated `idempotency_key`. Replays with identical parameters return the existing `ReservationRecord`. Replays with conflicting parameters raise `AIDuplicateSettlementError`.
- Every settlement requires an `idempotency_key`. Redeliveries (e.g. from Celery task retries) with identical settlement parameters return the existing `SettlementRecord`. Conflicting settlements raise `AIDuplicateSettlementError`.
- Duplicate settlement of an already finalized reservation with differing parameters is strictly prohibited fail-closed.

## 4. PostgreSQL / Durable Transaction & Fencing Requirements (Contract for M0-018B2)
When implementing the durable PostgreSQL repository in M0-018B2, the following requirements MUST be met:
1. **Row-level Locking**: `SELECT ... FOR UPDATE` on the budget settings and aggregate tracking rows within an isolated database transaction (`transaction.atomic`).
2. **Fencing Tokens / Attempt Numbers**: Each reservation carries a monotonically increasing `attempt_number`. If a worker attempts settlement with an `attempt_number < reservation.attempt_number`, the transaction must rollback and raise `AIStaleFencedAttemptError`.
3. **Unique Constraints**: Unique constraint on `(tenant_id, idempotency_key)` to enforce idempotency at the database storage layer.
4. **Append-only Ledger**: The `UsageLedgerEntry` table must be append-only with monotonic timestamps and sequence numbers.

## 5. Retry Behavior
- **Never-sent calls**: Safe to release via `release_reservation()`. Capacity is returned to the active budget without penalty.
- **Maybe-sent calls**: Must be settled as `SettlementType.ESTIMATED_EXPOSURE` or `SettlementType.UNKNOWN_USAGE`. Capacity is charged against estimated exposure until reconciled.

## 6. Privacy & Audit Contract
- No raw prompts, completion texts, provider API secret keys, authorization headers, or customer credentials may ever be stored in `ReservationRecord`, `SettlementRecord`, or `UsageLedgerEntry`.
- Only bounded audit metadata keys (e.g. `job_id`, `step_name`, `provider`, `model`), UTC timestamps, and numeric token tallies are persisted.
"""
