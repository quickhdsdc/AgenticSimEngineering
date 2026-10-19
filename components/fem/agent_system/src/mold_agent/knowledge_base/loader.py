"""知识库加载器: 读取 knowledge/*.yaml 供规则引擎使用。"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

# knowledge/ 目录: 项目根 (agent_system) 下
KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "knowledge"


def _load(name: str) -> dict:
    path = KNOWLEDGE_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def forming_rules() -> dict:
    return _load("forming_rules")


@cache
def optimization_rules() -> dict:
    return _load("optimization_rules")


@cache
def tolerance_rules() -> dict:
    return _load("tolerance_rules")


@cache
def process_rules() -> dict:
    return _load("process_rules")


@cache
def materials() -> dict:
    return _load("materials")


@cache
def report_template() -> dict:
    return _load("report_template")


def material_by_grade(grade: str) -> dict | None:
    g = (grade or "").strip().lower()
    for m in materials().get("materials", []):
        if m["grade"].strip().lower() == g:
            return m
    return None
