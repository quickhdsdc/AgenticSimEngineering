"""AutoForm 适配器: 用协议隔离真实 AutoForm 接口。

现状(见 docs/03): AutoForm R10 无 headless API。前期评估靠人工在 GUI 用模板搭好、
导出 QuickLink 包;本客户端负责"读结果"这一段(Agent 能完全接管)。
后续接入 Sigma/CLI 批量求解后,再让 run_forming 自动触发求解+导出。
"""
from __future__ import annotations

from typing import Protocol

from ..models import ProjectSpec, SimResult


class AutoFormClient(Protocol):
    def run_forming(self, package_or_spec) -> SimResult: ...


class QuickLinkAutoFormClient:
    """读取人工从 GUI 导出的 QuickLink 包(.zip)→ SimResult(真实数据,无任何拟合)。"""

    def __init__(self, package_path: str) -> None:
        self.package_path = package_path

    def run_forming(self, *_args, **_kwargs) -> SimResult:
        from .quicklink import parse_quicklink
        return parse_quicklink(self.package_path)
