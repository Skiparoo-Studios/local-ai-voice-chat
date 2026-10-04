"""Home Assistant, through its supported REST API.

The brief asks for the supported API rather than internal state, so this uses
``/api/states`` and ``/api/services``. Areas are not exposed by the states
endpoint, so they are fetched with a single template render, which is the
documented way to get them over REST; failure there degrades to no areas
rather than to no devices.

Generic actions map to services by translating the camelCase verb the assistant
uses into the snake_case Home Assistant expects: ``light.turnOff`` becomes a
POST to ``/api/services/light/turn_off``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar

from app.automation.automationProvider import (
    AutomationError,
    AutomationProvider,
    AutomationResult,
    AutomationUnavailableError,
    Device,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# Entity domains worth offering to the assistant. Home Assistant exposes a
# great many others (automations, scripts, updates) that would only add noise
# and risk to what the model can see.
CONTROLLABLE_DOMAINS = frozenset(
    {"light", "switch", "fan", "cover", "climate", "media_player", "lock", "scene"}
)
READABLE_DOMAINS = frozenset({"sensor", "binary_sensor", "weather"})

AREA_TEMPLATE = (
    "{% for state in states %}"
    "{{ state.entity_id }}|{{ area_name(state.entity_id) or '' }}\n"
    "{% endfor %}"
)


class HomeAssistantProvider(AutomationProvider):
    """Talks to a Home Assistant instance over HTTP."""

    name: ClassVar[str] = "home-assistant"

    def __init__(
        self,
        *,
        baseUrl: str,
        accessToken: str | None,
        timeoutSeconds: float = 15.0,
        transport: Any = None,
    ) -> None:
        if not baseUrl:
            raise AutomationError(
                "No Home Assistant address configured. Set automation.baseUrl, "
                "for example http://homeassistant.local:8123"
            )
        if not accessToken:
            raise AutomationError(
                "No Home Assistant token configured. Create a long-lived access "
                "token in your profile and set it as the VOICE_AUTOMATION__ACCESSTOKEN "
                "environment variable. It must not go in settings.json."
            )

        self._baseUrl = baseUrl.rstrip("/")
        self._accessToken = accessToken
        self._timeoutSeconds = timeoutSeconds
        self._transport = transport
        self._client: Any = None

    @classmethod
    def fromSettings(cls, settings: Settings) -> HomeAssistantProvider:
        section = settings.automation
        return cls(
            baseUrl=section.baseUrl,
            accessToken=(
                section.accessToken.get_secret_value() if section.accessToken else None
            ),
            timeoutSeconds=section.timeoutSeconds,
        )

    # --- Lifecycle -------------------------------------------------------

    async def load(self) -> None:
        self._ensureClient()

    def _ensureClient(self) -> Any:
        if self._client is None:
            httpx = _importHttpx()
            self._client = httpx.AsyncClient(
                base_url=self._baseUrl,
                timeout=self._timeoutSeconds,
                headers={
                    "Authorization": f"Bearer {self._accessToken}",
                    "Content-Type": "application/json",
                },
                transport=self._transport,
            )
        return self._client

    async def unload(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Devices ---------------------------------------------------------

    async def listDevices(self) -> list[Device]:
        client = self._ensureClient()

        try:
            response = await client.get("/api/states")
            response.raise_for_status()
            states = response.json()
        except Exception as error:
            raise _describeFailure(error, self._baseUrl) from error

        areas = await self._fetchAreas()
        devices: list[Device] = []

        for state in states:
            entityId = state.get("entity_id", "")
            domain = entityId.partition(".")[0]
            if domain not in CONTROLLABLE_DOMAINS and domain not in READABLE_DOMAINS:
                continue

            attributes = state.get("attributes") or {}
            devices.append(
                Device(
                    deviceId=entityId,
                    name=attributes.get("friendly_name") or _nameFromEntityId(entityId),
                    domain=domain,
                    area=areas.get(entityId),
                    state=state.get("state"),
                    attributes=attributes,
                )
            )

        logger.info("Home Assistant exposed %d usable devices", len(devices))
        return devices

    async def _fetchAreas(self) -> dict[str, str]:
        """Map entity ids to area names in one template render.

        Optional enrichment: without it devices still work, they are just
        harder to refer to by room.
        """
        client = self._ensureClient()
        try:
            response = await client.post("/api/template", json={"template": AREA_TEMPLATE})
            response.raise_for_status()
            rendered = response.text
        except Exception as error:  # noqa: BLE001 - areas are an enhancement
            logger.warning("Could not fetch areas from Home Assistant: %s", error)
            return {}

        areas: dict[str, str] = {}
        for line in rendered.splitlines():
            entityId, _, area = line.partition("|")
            entityId, area = entityId.strip(), area.strip()
            if entityId and area:
                areas[entityId] = area
        return areas

    # --- Actions ---------------------------------------------------------

    async def execute(self, action: str, parameters: dict[str, Any]) -> AutomationResult:
        domain, service = translateAction(action)
        target = str(parameters.get("target") or "")

        if not target:
            return AutomationResult.failed(action, target, "No target device was given")

        payload: dict[str, Any] = {"entity_id": target}
        for key, value in parameters.items():
            if key not in {"target", "action"} and value is not None:
                payload[_toSnakeCase(key)] = value

        client = self._ensureClient()
        try:
            response = await client.post(f"/api/services/{domain}/{service}", json=payload)
            response.raise_for_status()
            changed = response.json()
        except Exception as error:
            raise _describeFailure(error, self._baseUrl) from error

        logger.info("Called %s/%s on %s", domain, service, target)
        return AutomationResult.ok(
            action=action,
            target=target,
            message=f"{action} sent to {target}",
            changedEntities=[entry.get("entity_id") for entry in changed or []],
        )

    async def isAvailable(self) -> bool:
        try:
            client = self._ensureClient()
            response = await client.get("/api/", timeout=5.0)
            return response.status_code == 200
        except Exception as error:  # noqa: BLE001 - availability is a question
            logger.debug("Home Assistant not reachable: %s", error)
            return False

    def describe(self) -> str:
        return f"home-assistant at {self._baseUrl}"


# --- Module helpers ------------------------------------------------------


def translateAction(action: str) -> tuple[str, str]:
    """Turn a generic ``domain.verbName`` action into a Home Assistant service."""
    domain, _, verb = action.partition(".")
    if not domain or not verb:
        raise AutomationError(
            f"Action {action!r} is not in the expected 'domain.verb' form, "
            "for example 'light.turnOff'."
        )
    return domain, _toSnakeCase(verb)


def _toSnakeCase(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _nameFromEntityId(entityId: str) -> str:
    return entityId.partition(".")[2].replace("_", " ").strip() or entityId


def _importHttpx() -> Any:
    try:
        import httpx
    except ImportError as error:
        raise AutomationUnavailableError(
            "Home Assistant support requires httpx.\n"
            'Install with: pip install -e ".[automation]"'
        ) from error
    return httpx


def _describeFailure(error: Exception, baseUrl: str) -> Exception:
    """Translate transport failures into messages that say what to do."""
    text = str(error).lower()
    status = getattr(getattr(error, "response", None), "status_code", None)

    if status in (401, 403):
        return AutomationUnavailableError(
            f"Home Assistant at {baseUrl} rejected the access token.\n"
            "Create a long-lived access token in your profile and set it as "
            "VOICE_AUTOMATION__ACCESSTOKEN."
        )

    if status == 404:
        return AutomationUnavailableError(
            f"Home Assistant at {baseUrl} did not recognise that request. "
            "Check automation.baseUrl points at the instance root, not a sub-path."
        )

    if isinstance(error, AutomationError):
        return error

    if "connect" in text or "refused" in text or "timed out" in text:
        return AutomationUnavailableError(
            f"Could not reach Home Assistant at {baseUrl}.\n"
            "Check the address and that this machine can see it.\n\n"
            f"Original error: {error}"
        )

    return AutomationError(f"Home Assistant request failed: {error}")


__all__ = ["HomeAssistantProvider", "translateAction"]
