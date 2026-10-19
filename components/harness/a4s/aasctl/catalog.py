"""Operational discovery, distinct from raw repository/admin access.

AssetInformation.assetKind (Instance/Type) and Submodel.kind
(Instance/Template) are different axes. No naming heuristic is used.
"""

import os


# The existing ShopFloor explicitly instantiates from this shell, although the
# deployed legacy model calls it Instance. Hide that configured blueprint from
# normal operational discovery without rewriting anyone's repository metadata.
LEGACY_BLUEPRINTS = {"https://fraunhoferIPA/ids/aas/FestoWorkCenterStamping"}


def is_blueprint(shell):
    configured = {
        value.strip() for value in os.getenv("A4S_BLUEPRINT_IDS", "").split(",") if value.strip()
    }
    return (
        (shell.get("assetInformation") or {}).get("assetKind") == "Type"
        or shell.get("id") in LEGACY_BLUEPRINTS | configured
    )


def shells_in_view(shells, kind="Instance"):
    if kind == "all":
        return list(shells)
    if kind == "Type":
        return [shell for shell in shells if is_blueprint(shell)]
    return [
        shell for shell in shells
        if (shell.get("assetInformation") or {}).get("assetKind") == kind
        and not is_blueprint(shell)
    ]


def submodels_in_view(submodels, shells=(), kind="Instance"):
    if kind == "all":
        return list(submodels)
    operational_refs, blueprint_refs = set(), set()
    for shell in shells:
        refs = blueprint_refs if is_blueprint(shell) else operational_refs
        refs.update(
            key.get("value") for ref in shell.get("submodels") or []
            for key in ref.get("keys") or [] if key.get("type") == "Submodel"
        )
    blueprint_only = blueprint_refs - operational_refs
    return [
        sm for sm in submodels
        if (sm.get("kind") or "Instance") == kind
        and (kind != "Instance" or sm.get("id") not in blueprint_only)
    ]
