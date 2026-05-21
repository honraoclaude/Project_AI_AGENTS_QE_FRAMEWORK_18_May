from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://qe_agent_writer:writer_localdev@localhost:5432/qe_framework"
    database_admin_url: str = "postgresql+psycopg://qe_admin:localdev@localhost:5432/qe_framework"
    database_reader_url: str = "postgresql+psycopg://qe_audit_reader:reader_localdev@localhost:5432/qe_framework"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Anthropic
    anthropic_api_key: SecretStr = SecretStr("")

    # Model selection
    default_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"
    confidence_escalation_threshold: int = 60

    # Jira
    jira_url: str = ""
    jira_username: str = ""
    jira_api_token: SecretStr = SecretStr("")
    jira_project_key: str = "FSC"
    jira_ac_field: str = "customfield_10200"

    # Salesforce
    sf_username: str = ""
    sf_password: SecretStr = SecretStr("")
    sf_security_token: SecretStr = SecretStr("")
    sf_domain: str = "test"

    # Copado (Development phase — branch tracing, coverage, metadata)
    copado_url: str = ""
    copado_access_token: SecretStr = SecretStr("")

    # Sign-off service
    signoff_base_url: str = "http://localhost:8000"
    signoff_hmac_secret: SecretStr = SecretStr("change-me")
    signoff_link_expiry_hours: int = 48

    # Azure Communication Services
    azure_comm_connection_string: SecretStr = SecretStr("")
    azure_comm_sender: str = "qe-framework@firm.internal"

    # Compliance contacts
    compliance_officer_email: str = ""
    qe_lead_email: str = ""
    tech_lead_email: str = ""


settings = Settings()
