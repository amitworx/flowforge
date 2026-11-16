"""Django AppConfig for FlowForge AI Governance persistence layer."""

from django.apps import AppConfig


class AIGovernanceDBConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "flowforge.ai.governance.db"
    label = "ai_governance"
    verbose_name = "AI Governance Persistence"
