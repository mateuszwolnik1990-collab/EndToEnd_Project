"""Testy bramek bezpieczeństwa i sterowania: Stop, blokada MVP bez zgody, brak uruchamiania kodu."""

from __future__ import annotations

import pytest

from app_discovery_system.config import DEFAULT_SAFETY_CONFIG, VLLMConfig
from app_discovery_system.llm.vllm_client import CampaignStoppedError
from app_discovery_system.models.schemas import MVPSpecification
from app_discovery_system.orchestrator.campaign_runner import CampaignController
from app_discovery_system.tools.verification_tools import (
    static_code_audit,
    write_mvp_files_safely,
)


@pytest.fixture()
def patched_db(temp_db, monkeypatch):
    import app_discovery_system.orchestrator.campaign_runner as runner

    monkeypatch.setattr(runner, "DatabaseManager", lambda *a, **k: temp_db)
    return temp_db


def test_safety_config_forbids_side_effects():
    assert DEFAULT_SAFETY_CONFIG.allow_external_publishing is False
    assert DEFAULT_SAFETY_CONFIG.allow_purchases is False
    assert DEFAULT_SAFETY_CONFIG.allow_sending_messages is False
    assert DEFAULT_SAFETY_CONFIG.allow_unverified_code_execution is False
    assert DEFAULT_SAFETY_CONFIG.require_user_approval_for_mvp is True


def test_stop_flag_is_detected_in_database(temp_db):
    campaign_id = temp_db.create_campaign("Stop test")
    assert temp_db.is_stop_requested(campaign_id) is False
    temp_db.update_campaign_status(campaign_id, "STOPPED")
    assert temp_db.is_stop_requested(campaign_id) is True


def test_stop_callback_aborts_llm_call(temp_db, monkeypatch):
    import app_discovery_system.orchestrator.campaign_runner as runner

    monkeypatch.setattr(runner, "DatabaseManager", lambda *a, **k: temp_db)
    campaign_id = temp_db.create_campaign("Stop callback")
    temp_db.update_campaign_status(campaign_id, "STOPPED")

    llm, _ = runner.build_llm(campaign_id, VLLMConfig(), use_simulator=True)
    with pytest.raises(CampaignStoppedError):
        llm.call(messages=[{"role": "user", "content": "[TASK:IDEAS]"}])  # type: ignore[attr-defined]


def test_controller_stop_sets_status(temp_db, monkeypatch):
    import app_discovery_system.orchestrator.campaign_runner as runner

    monkeypatch.setattr(runner, "DatabaseManager", lambda *a, **k: temp_db)
    controller = CampaignController(vllm_config=VLLMConfig(), db=temp_db)
    campaign_id = temp_db.create_campaign("Ctrl")
    controller.stop_campaign(campaign_id)
    assert temp_db.get_campaign(campaign_id)["status"] == "STOPPED"


def test_approve_idea_gate(patched_db):
    db = patched_db
    campaign_id = db.create_campaign("Gate")
    from app_discovery_system.orchestrator.campaign_runner import _execute_discovery

    _execute_discovery(campaign_id, VLLMConfig(), use_simulator=True)
    top = [i for i in db.get_ideas_for_campaign(campaign_id) if i["is_top3"]][0]
    assert top["approved_for_mvp"] is False

    controller = CampaignController(vllm_config=VLLMConfig(), db=db)
    controller.approve_idea(campaign_id, top["id"])
    assert db.get_idea_by_id(top["id"])["approved_for_mvp"] is True


def test_build_mvp_blocked_without_approval(patched_db):
    db = patched_db
    campaign_id = db.create_campaign("Blocked")
    from app_discovery_system.orchestrator.campaign_runner import _execute_discovery

    _execute_discovery(campaign_id, VLLMConfig(), use_simulator=True)
    top = [i for i in db.get_ideas_for_campaign(campaign_id) if i["is_top3"]][0]

    controller = CampaignController(vllm_config=VLLMConfig(), db=db)
    with pytest.raises(PermissionError):
        controller.build_mvp(campaign_id, top["id"], use_simulator=True)


def test_static_audit_flags_dangerous_code():
    safe_files = {"app/main.py": "import fastapi\n\ndef health():\n    return 1\n"}
    is_safe, report = static_code_audit(safe_files)
    assert is_safe is True
    assert "OK" in report

    unsafe_files = {"bad.py": "import subprocess\nsubprocess.run(['ls'])\n"}
    is_safe2, report2 = static_code_audit(unsafe_files)
    assert is_safe2 is False
    assert "subprocess" in report2


def test_write_mvp_files_blocks_path_traversal(tmp_path):
    files = {"../escape.py": "print(1)", "app/ok.py": "print(2)"}
    sandbox, written = write_mvp_files_safely("demo", files, base_dir=tmp_path)
    assert sandbox == tmp_path / "demo"
    assert (sandbox / "app" / "ok.py").exists()
    assert not (tmp_path / "escape.py").exists()
    assert len(written) == 1


def test_mvp_specification_model_roundtrip():
    spec = MVPSpecification(
        idea_title="X",
        category="B2B",
        architecture_overview="a",
        tech_stack=["Python"],
        core_features=["f"],
        database_schema_sql="CREATE TABLE t(id INT);",
        api_endpoints=["GET /health"],
        generated_files={"README.md": "# X"},
        qa_test_plan="p",
        static_security_audit="s",
        safe_run_instructions="i",
    )
    assert spec.generated_files["README.md"] == "# X"
