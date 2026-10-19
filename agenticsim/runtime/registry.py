"""Explicit allowlist; no task-supplied Python import paths."""
from agenticsim.domain import require


def evaluator_for(document):
    require(isinstance(document, dict), "task must be an object")
    model = document.get("model", {})
    require(isinstance(model, dict), "model must be an object")
    if model.get("id") == "fem-sample7":
        from agenticsim.adapters.fem import StampingFEM
        return StampingFEM()
    if model.get("id") == "fifo-single-server":
        from agenticsim.adapters.des import SingleServer
        return SingleServer()
    if model.get("id") == "planar-arm3":
        from agenticsim.adapters.newton_robot import NewtonRobot
        return NewtonRobot()
    if model.get("id") in ("kuka-iiwa14", "unitree-g1"):
        from agenticsim.adapters.asset_robot import AssetRobot
        return AssetRobot(model["id"])
    raise ValueError("unregistered model")
