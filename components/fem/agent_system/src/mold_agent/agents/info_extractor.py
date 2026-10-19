"""信息提取 Agent (MVP 版)。

现状: 从报价单/顾客需求表等 Excel 抽取结构化 ProjectSpec。
- 已实现: 读 .xls/.xlsx 关键字段的启发式抽取。
- TODO(M2): STP 几何解析、LLM 读 2D 图纸公差。
"""
from __future__ import annotations

from pathlib import Path

from ..knowledge_base import loader
from ..models import ProjectSpec


def _find_material(text: str) -> str:
    for m in loader.materials()["materials"]:
        if m["grade"].replace(" ", "").lower() in text.replace(" ", "").lower():
            return m["grade"]
    return ""


def extract_from_folder(folder: str | Path, fallback: dict | None = None) -> ProjectSpec:
    """从项目文件夹抽取 ProjectSpec。

    MVP: 仅登记文件清单 + 用 fallback/文件名做最小抽取。
    真实抽取(读 Excel 单元格、STP、图纸)在 M2 接入;此处保留接口。
    """
    folder = Path(folder)
    files = [p.name for p in folder.rglob("*") if p.is_file()] if folder.exists() else []
    spec = ProjectSpec(source_files=files)

    if fallback:
        spec = spec.model_copy(update=fallback)
    if not spec.material and files:
        spec.material = _find_material(" ".join(files))
    return spec


def extract_from_dict(data: dict) -> ProjectSpec:
    """直接从已知参数构造 (用于 demo / 已人工录入的场景)。"""
    return ProjectSpec(**data)
