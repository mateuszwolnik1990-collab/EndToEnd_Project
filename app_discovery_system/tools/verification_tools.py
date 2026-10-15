"""Narzędzia weryfikacji i bezpieczeństwa.

- ``verify_claims``: waliduje listę twierdzeń rynkowych i wymusza regułę "źródło albo [HIPOTEZA]".
- ``write_mvp_files_safely``: zapisuje pliki wygenerowanego MVP wyłącznie w sandboxie (bez uruchamiania).
- ``static_code_audit``: statyczna analiza kodu MVP (AST) wykrywająca niebezpieczne wywołania.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app_discovery_system.config import MVP_WORKSPACE_DIR
from app_discovery_system.models.schemas import ClaimClassification, MarketClaim

# Wzorce niebezpiecznych operacji, których nie wolno uruchamiać bez zgody użytkownika.
_DANGEROUS_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "system",
    "popen",
    "spawn",
    "fork",
    "remove",
    "rmtree",
    "unlink",
    "rmdir",
}
_DANGEROUS_MODULES = {"subprocess", "shutil", "ctypes", "pty", "socket", "multiprocessing"}


def validate_claims(raw_claims: List[Dict[str, Any]]) -> List[MarketClaim]:
    """Zamienia surowe słowniki na ``MarketClaim``, wymuszając źródło URL lub [HIPOTEZA]."""
    claims: List[MarketClaim] = []
    for raw in raw_claims or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("claim_text") or raw.get("text") or "").strip()
        if not text:
            continue
        claims.append(
            MarketClaim(
                claim_text=text,
                source_url=raw.get("source_url") or raw.get("url"),
                classification=ClaimClassification.HYPOTHESIS,
            )
        )
    return claims


def _is_within(base: Path, target: Path) -> bool:
    """Sprawdza, czy ścieżka docelowa nie wychodzi poza katalog sandbox (ochrona przed traversal)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def write_mvp_files_safely(
    idea_slug: str, files: Dict[str, str], base_dir: Path | None = None
) -> Tuple[Path, List[str]]:
    """Zapisuje pliki MVP w izolowanym katalogu. NIGDY nie uruchamia kodu.

    Zwraca krotkę (katalog_sandbox, lista_zapisanych_ścieżek).
    """
    base = Path(base_dir) if base_dir else MVP_WORKSPACE_DIR
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", idea_slug).strip("_") or "mvp"
    sandbox = base / safe_slug
    sandbox.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for name, content in (files or {}).items():
        if not isinstance(name, str) or not isinstance(content, str):
            continue
        rel = Path(name.lstrip("/\\"))
        target = sandbox / rel
        if not _is_within(sandbox, target):
            continue  # Odrzucamy próby wyjścia poza sandbox (path traversal).
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(str(target))
    return sandbox, written


def static_code_audit(files: Dict[str, str]) -> Tuple[bool, str]:
    """Statyczna analiza kodu (AST) — wykrywa niebezpieczne wywołania bez jego uruchamiania.

    Zwraca krotkę (is_safe, raport).
    """
    findings: List[str] = []
    python_files = 0
    for name, content in (files or {}).items():
        if not isinstance(name, str) or not isinstance(content, str):
            continue
        if not name.endswith(".py"):
            continue
        python_files += 1
        try:
            tree = ast.parse(content, filename=name)
        except SyntaxError as exc:
            findings.append(f"{name}: BŁĄD SKŁADNI -> {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = getattr(func, "id", None) or getattr(func, "attr", None)
                if func_name in _DANGEROUS_CALLS:
                    findings.append(f"{name}: niebezpieczne wywołanie '{func_name}' (linia {node.lineno})")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _DANGEROUS_MODULES:
                        findings.append(f"{name}: import modułu systemowego '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _DANGEROUS_MODULES:
                    findings.append(f"{name}: import z modułu systemowego '{node.module}'")

    if findings:
        report = (
            f"Audyt statyczny: WYKRYTO {len(findings)} potencjalnych zagrożeń w {python_files} plikach .py.\n"
            + "\n".join(f"- {f}" for f in findings)
            + "\nKod NIE może być uruchomiony bez ręcznej weryfikacji użytkownika."
        )
        return False, report

    return (
        True,
        f"Audyt statyczny: OK. Przeanalizowano {python_files} plików .py. "
        "Nie wykryto niebezpiecznych wywołań (eval/exec/subprocess/os.system) ani importów modułów systemowych.",
    )
