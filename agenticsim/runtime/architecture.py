"""Check import direction without importing solvers or optional SDKs."""
import ast
from importlib.util import resolve_name
from pathlib import Path
import sys

CORE_ALLOWED = {
    "domain": {"domain"},
    "ports": {"domain", "ports"},
    "application": {"domain", "ports", "application"},
}
HEAVY = {"torch", "julia", "newton", "mujoco", "warp", "isaacsim", "omni", "pxr", "pychrono", "openai", "anthropic"}


def violations(source, module, *, layer=None):
    findings = []
    tree = ast.parse(source)
    package = module if module.endswith(".__init__") else module.rpartition(".")[0]
    package = package.removesuffix(".__init__")
    module_level = set()

    def at_module(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_level.add(id(node))
            for child in ast.iter_child_nodes(node):
                at_module([child])
    at_module(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = resolve_name("." * node.level + name, package)
            names = [name]
        else:
            continue
        for name in names:
            first = name.split(".")[0]
            reason = None
            if layer in CORE_ALLOWED:
                if first == "agenticsim":
                    dependency = name.split(".")[1] if "." in name else ""
                    if dependency not in CORE_ALLOWED[layer]:
                        reason = "core dependency points outside its permitted layers"
                elif first not in sys.stdlib_module_names:
                    reason = "core imports an optional or external dependency"
            elif id(node) in module_level and first in HEAVY:
                reason = "heavy SDK must be imported at execution time"
            if reason:
                findings.append({"module": module, "line": node.lineno, "import": name, "reason": reason})
    return findings


def check(root):
    findings = []
    for path in sorted((Path(root) / "agenticsim").rglob("*.py")):
        relative = path.relative_to(root).with_suffix("")
        module = ".".join(relative.parts)
        layer = relative.parts[1] if len(relative.parts) > 2 else None
        findings.extend(violations(path.read_text(encoding="utf-8"), module, layer=layer))
    return findings
