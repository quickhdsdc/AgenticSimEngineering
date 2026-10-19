"""Dependency inversion boundary: core services know no solver SDK."""
from typing import Protocol
from agenticsim.domain import BackendDescriptor, EngineeringTask, Evaluation, ModelDefinition


class Evaluator(Protocol):
    model: ModelDefinition
    backend: BackendDescriptor

    def validate(self, task: EngineeringTask) -> None: ...
    def evaluate(self, task: EngineeringTask, candidate: dict) -> Evaluation: ...


class Executor(Protocol):
    def evaluate(self, task: EngineeringTask, candidate: dict) -> Evaluation: ...


class StudyStore(Protocol):
    def create(self, task: dict) -> dict: ...
    def save(self, study: dict) -> None: ...
