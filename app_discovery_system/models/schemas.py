"""Modele Pydantic dla twierdzeń rynkowych, oceny 0-100, pomysłów, raportu Top 3 i specyfikacji MVP."""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class AppCategory(str, Enum):
    MOBILE = "Mobile"
    WEB = "Web"
    B2B = "B2B"


class ClaimClassification(str, Enum):
    VERIFIED_SOURCE = "VERIFIED_SOURCE"
    HYPOTHESIS = "HYPOTHESIS"


def is_valid_http_url(url: Optional[str]) -> bool:
    """Sprawdza, czy podany ciąg jest poprawnym adresem URL HTTP/HTTPS."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc) and "." in parsed.netloc
    except Exception:
        return False


class MarketClaim(BaseModel):
    """Pojedyncze twierdzenie rynkowe. Musi mieć źródło URL lub być oznaczone jako [HIPOTEZA]."""

    claim_text: str = Field(..., min_length=3, description="Treść twierdzenia rynkowego")
    source_url: Optional[str] = Field(default=None, description="URL rzeczywistego źródła")
    classification: ClaimClassification = Field(
        default=ClaimClassification.HYPOTHESIS,
        description="VERIFIED_SOURCE (ze źródłem) lub HYPOTHESIS ([HIPOTEZA])",
    )

    @model_validator(mode="after")
    def enforce_source_or_hypothesis(self) -> "MarketClaim":
        """Brak poprawnego URL automatycznie oznacza twierdzenie jako [HIPOTEZA]."""
        if is_valid_http_url(self.source_url):
            self.classification = ClaimClassification.VERIFIED_SOURCE
            self.source_url = self.source_url.strip()  # type: ignore[union-attr]
            self.claim_text = re.sub(r"\s*\[HIPOTEZA\]\s*", " ", self.claim_text).strip()
        else:
            self.classification = ClaimClassification.HYPOTHESIS
            self.source_url = None
            if "[HIPOTEZA]" not in self.claim_text:
                self.claim_text = f"{self.claim_text.strip()} [HIPOTEZA]"
        return self

    def formatted(self) -> str:
        """Zwraca sformatowane twierdzenie ze źródłem lub oznaczeniem [HIPOTEZA]."""
        if self.classification == ClaimClassification.VERIFIED_SOURCE and self.source_url:
            return f"{self.claim_text} [ŹRÓDŁO: {self.source_url}]"
        if "[HIPOTEZA]" not in self.claim_text:
            return f"{self.claim_text} [HIPOTEZA]"
        return self.claim_text


class ScoreBreakdown(BaseModel):
    """Szczegółowa punktacja pomysłu składająca się na ocenę 0-100."""

    market_demand: int = Field(..., ge=0, le=25)
    monetization_potential: int = Field(..., ge=0, le=25)
    low_competition_niche: int = Field(..., ge=0, le=20)
    build_feasibility: int = Field(..., ge=0, le=15)
    sales_channel_access: int = Field(..., ge=0, le=15)

    @property
    def total_score(self) -> int:
        total = (
            self.market_demand
            + self.monetization_potential
            + self.low_competition_niche
            + self.build_feasibility
            + self.sales_channel_access
        )
        return max(0, min(100, total))


class AppIdea(BaseModel):
    """Model pojedynczego pomysłu na aplikację (z puli 10 pomysłów oraz rozszerzonej analizy dla Top 3)."""

    rank: int = Field(default=1, ge=1, le=100)
    title: str = Field(..., min_length=3)
    category: AppCategory = Field(...)
    problem_statement: str = Field(..., min_length=10)
    target_audience: str = Field(..., min_length=5)
    value_proposition: str = Field(..., min_length=10)
    score_breakdown: ScoreBreakdown
    score_total: int = Field(default=0, ge=0, le=100)
    is_top3: bool = Field(default=False)
    claims: List[MarketClaim] = Field(default_factory=list)

    competitors_analysis: str = Field(default="")
    pricing_analysis: str = Field(default="")
    build_difficulty: str = Field(default="")
    sales_channels: str = Field(default="")
    recommendation: str = Field(default="")

    @model_validator(mode="after")
    def sync_total_score_and_claims(self) -> "AppIdea":
        self.score_total = self.score_breakdown.total_score
        if not self.claims:
            self.claims = [
                MarketClaim(
                    claim_text=self.problem_statement,
                    source_url=None,
                    classification=ClaimClassification.HYPOTHESIS,
                )
            ]
        return self


class CampaignReport(BaseModel):
    """Kompletny wynik Kampanii Odkrywania (10 pomysłów + głęboka analiza 3 najlepszych + rekomendacja CEO)."""

    campaign_name: str
    ideas: List[AppIdea] = Field(..., min_length=10, max_length=10)
    top3_titles: List[str] = Field(..., min_length=3, max_length=3)
    executive_summary: str
    final_recommendation: str

    @field_validator("ideas")
    @classmethod
    def check_categories_diversity(cls, ideas: List[AppIdea]) -> List[AppIdea]:
        categories = {idea.category for idea in ideas}
        required = {AppCategory.MOBILE, AppCategory.WEB, AppCategory.B2B}
        if not required.issubset(categories):
            missing = required - categories
            raise ValueError(f"Lista 10 pomysłów musi obejmować Mobile, Web i B2B. Brakuje: {missing}")
        return ideas


class MVPSpecification(BaseModel):
    """Specyfikacja MVP oraz moduł wygenerowanego (nieuruchamianego automatycznie) kodu aplikacji."""

    idea_title: str
    category: AppCategory
    architecture_overview: str
    tech_stack: List[str]
    core_features: List[str]
    database_schema_sql: str
    api_endpoints: List[str]
    generated_files: Dict[str, str] = Field(default_factory=dict)
    qa_test_plan: str
    static_security_audit: str
    safe_run_instructions: str
