"""The tool interface.

Per §11 of the brief, every tool declares a name, a description, an input
schema and an execute method, so that deterministic code and a language model
can invoke the same capability.

The classification that matters most is ``mutatesPhysicalState``. §17 requires
read-only tools to be separated from those that change the world, because the
consequences of a mistaken call differ enormously between them: reading a
temperature wrongly is a wrong answer, unlocking a door wrongly is a break-in.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Raised when a tool cannot run."""


class ToolValidationError(ToolError):
    """Raised when arguments do not satisfy a tool's schema."""


class ToolNotPermittedError(ToolError):
    """Raised when a tool or action is not on the allow-list."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of running a tool."""

    succeeded: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    # Whether the message is ready to be spoken as it stands. A tool that
    # returns raw data sets this false so the caller can phrase it.
    speakable: bool = True

    @classmethod
    def ok(cls, message: str, **details: Any) -> ToolResult:
        return cls(succeeded=True, message=message, details=details)

    @classmethod
    def failed(cls, message: str, **details: Any) -> ToolResult:
        return cls(succeeded=False, message=message, details=details)


class Tool(ABC):
    """A capability the assistant can invoke."""

    name: ClassVar[str] = "unnamed"
    description: ClassVar[str] = ""
    inputModel: ClassVar[type[BaseModel]]
    # False for anything that changes the physical environment.
    readOnly: ClassVar[bool] = False
    # Requires explicit confirmation before running, regardless of allow-list.
    requiresConfirmation: ClassVar[bool] = False

    @abstractmethod
    async def execute(self, parameters: BaseModel) -> ToolResult:
        """Run the tool with validated parameters."""

    @property
    def mutatesPhysicalState(self) -> bool:
        return not self.readOnly

    def validate(self, rawParameters: dict[str, Any]) -> BaseModel:
        """Turn untrusted arguments into a validated model.

        Language model output reaches this method, so failures are expected
        rather than exceptional and produce a message the model can act on.
        """
        try:
            return self.inputModel.model_validate(rawParameters or {})
        except ValidationError as error:
            raise ToolValidationError(
                f"Arguments for {self.name!r} are not valid: {_summariseErrors(error)}"
            ) from error

    def toSchema(self) -> dict[str, Any]:
        """Describe the tool the way function-calling APIs expect."""
        schema = self.inputModel.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def describe(self) -> str:
        kind = "read-only" if self.readOnly else "changes state"
        return f"{self.name} ({kind})"


def _summariseErrors(error: ValidationError) -> str:
    """Render validation failures briefly enough to speak or send to a model."""
    parts = []
    for item in error.errors():
        location = ".".join(str(piece) for piece in item.get("loc", ())) or "input"
        parts.append(f"{location}: {item.get('msg', 'invalid')}")
    return "; ".join(parts[:3])
