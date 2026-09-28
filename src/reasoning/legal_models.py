from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


SCHEMA_VERSION = "1"


def _require_text_or_children(text: str | None, has_children: bool) -> None:
    if text is None and has_children:
        return
    if text and text.strip():
        return
    raise ValueError("text must contain non-whitespace characters")


def _require_unique(items: list[BaseModel], field_name: str) -> None:
    values: set[str] = set()
    for item in items:
        value = getattr(item, field_name)
        if value in values:
            raise ValueError(f"duplicate {field_name}: {value}")
        values.add(value)


class LegalSource(BaseModel):
    source_url: str
    raw_file: str
    sha256: str
    retrieved_at: datetime


class LegalPoint(BaseModel):
    point_id: str
    text: str | None = None

    @model_validator(mode="after")
    def validate_text(self) -> LegalPoint:
        _require_text_or_children(self.text, has_children=False)
        return self


class LegalClause(BaseModel):
    clause_id: str
    text: str | None = None
    points: list[LegalPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> LegalClause:
        _require_text_or_children(self.text, has_children=bool(self.points))
        _require_unique(self.points, "point_id")
        return self


class LegalArticle(BaseModel):
    article_id: str
    text: str | None = None
    clauses: list[LegalClause] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> LegalArticle:
        _require_text_or_children(self.text, has_children=bool(self.clauses))
        _require_unique(self.clauses, "clause_id")
        return self


class LegalSection(BaseModel):
    section_id: str
    text: str | None = None
    articles: list[LegalArticle] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> LegalSection:
        _require_text_or_children(self.text, has_children=bool(self.articles))
        _require_unique(self.articles, "article_id")
        return self


class LegalDocument(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    document_id: str
    title: str
    document_type: str
    source: LegalSource
    articles: list[LegalArticle] = Field(default_factory=list)
    sections: list[LegalSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> LegalDocument:
        _require_unique(self.articles, "article_id")
        _require_unique(self.sections, "section_id")
        return self
