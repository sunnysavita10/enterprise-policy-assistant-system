from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    policy_dir: Path = data_dir / "policies"
    upload_dir: Path = data_dir / "uploads"
    chroma_dir: Path = project_root / "chroma_db"
    collection_name: str = os.getenv("CHROMA_COLLECTION", "enterprise-policy-assistant")

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    enable_llm_injection_check: bool = os.getenv("ENABLE_LLM_INJECTION_CHECK", "true").lower() == "true"
    enable_groundedness_check: bool = os.getenv("ENABLE_GROUNDEDNESS_CHECK", "true").lower() == "true"
    groundedness_threshold: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.75"))

    @property
    def all_document_dirs(self) -> list[Path]:
        return [self.policy_dir, self.upload_dir]


settings = Settings()
settings.policy_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
