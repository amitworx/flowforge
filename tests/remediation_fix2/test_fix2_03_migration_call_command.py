"""Migration test using call_command for rollback and forward migration verification."""

import pytest
from django.core.management import call_command
from django.db import connection

from flowforge.ai.governance.db.models import AIReservationModel
from flowforge.ai.governance.models import DispatchStatus, ReservationStatus


@pytest.mark.django_db(transaction=True)
def test_fix2_03_migration_upgrade_and_rollback_flow():
    """Verify that migration rollback to 0001 and forward re-apply safely fences historical rows."""
    # 1. Rollback to 0001
    call_command("migrate", "ai_governance", "0001", verbosity=0)

    # 2. Insert historical raw reservation directly into ai_governance_reservation table (which has no dispatch_status in 0001)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, created_at, audit_metadata
            ) VALUES (
                'hist-callcmd-001', 'hist-callcmd-idem-1', 'tenant:corp-mig', 'tenant', 'corp-mig',
                'mock', 'm1', 100, 100, 200, 1000, 1,
                'pending', NOW(), '{}'::jsonb
            );
            """
        )
        cursor.execute(
            """
            INSERT INTO ai_governance_reservation (
                reservation_id, idempotency_key, scope_key, scope_type, tenant_id,
                provider, model, estimated_prompt_tokens, estimated_completion_tokens,
                estimated_total_tokens, estimated_cost_micros, attempt_number,
                status, created_at, audit_metadata
            ) VALUES (
                'hist-callcmd-settled', 'hist-callcmd-idem-2', 'tenant:corp-mig', 'tenant', 'corp-mig',
                'mock', 'm1', 100, 100, 200, 1000, 1,
                'settled', NOW(), '{}'::jsonb
            );
            """
        )

    # 3. Migrate forward to latest (0002 -> 0003 -> 0004 -> 0005)
    call_command("migrate", "ai_governance", verbosity=0)

    # 4. Check historical pending reservation status
    hist_pending = AIReservationModel.objects.get(reservation_id="hist-callcmd-001")
    assert hist_pending.dispatch_status == DispatchStatus.BILLING_UNCERTAIN.value, (
        f"Expected historical pending reservation to be fenced to billing_uncertain, got {hist_pending.dispatch_status}"
    )

    # Check settled reservation
    hist_settled = AIReservationModel.objects.get(reservation_id="hist-callcmd-settled")
    assert hist_settled.status == ReservationStatus.SETTLED.value
