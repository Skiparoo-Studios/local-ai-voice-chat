"""The tool registry, and the boundary every invocation crosses.

Everything that reaches a tool passes through :meth:`ToolRegistry.invoke`,
whether it came from a deterministic pattern or from a language model. That is
the point: §17 requires every invocation to be validated, and one path is far
easier to reason about than two.

The registry enforces *tool-level* policy --- which tools exist, and whether
state-changing tools may run at all. It deliberately does not enforce the
automation *action* allow-list, which lives with the automation layer in
:mod:`app.tools.homeAutomation`. Putting "light.turnOff is permitted" in a
generic registry would mean the registry understood home automation semantics,
which is the coupling §10 exists to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.tools.base import (
    Tool,
    ToolError,
    ToolNotPermittedError,
    ToolResult,
    ToolValidationError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A request to run a tool, as proposed by code or by a model."""

    name: str
    arguments: dict[str, Any]


class ToolRegistry:
    """Holds the available tools and mediates access to them."""

    def __init__(self, *, allowStateChanges: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        self._allowStateChanges = allowStateChanges

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"A tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        logger.debug("Registered tool %s", tool.describe())

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotPermittedError(
                f"No tool named {name!r}. Available: {known}"
            ) from None

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self, *, readOnlyOnly: bool = False) -> list[dict[str, Any]]:
        """Tool descriptions for a function-calling model.

        A model cannot invoke what it cannot see, so restricting this list is
        the first and cheapest control on what it can reach.
        """
        return [
            tool.toSchema()
            for tool in self._tools.values()
            if not readOnlyOnly or tool.readOnly
        ]

    async def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Validate and run a tool. The only way tools are executed."""
        tool = self.get(name)

        if tool.mutatesPhysicalState and not self._allowStateChanges:
            logger.warning("Refused %r: state changes are disabled", name)
            raise ToolNotPermittedError(
                f"{name!r} changes the physical environment, which is currently disabled."
            )

        parameters = tool.validate(arguments or {})

        logger.info(
            "Invoking %s%s", name, " (changes state)" if tool.mutatesPhysicalState else ""
        )
        try:
            return await tool.execute(parameters)
        except ToolError:
            raise
        except Exception as error:
            logger.exception("Tool %s failed", name)
            raise ToolError(f"{name!r} failed: {error}") from error

    def describe(self) -> str:
        if not self._tools:
            return "no tools registered"
        return ", ".join(tool.describe() for tool in self._tools.values())


__all__ = [
    "ToolInvocation",
    "ToolNotPermittedError",
    "ToolRegistry",
    "ToolResult",
    "ToolValidationError",
]
