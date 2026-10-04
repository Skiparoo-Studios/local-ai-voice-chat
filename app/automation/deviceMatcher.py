"""Resolving spoken device names to entity identifiers.

People say "the kitchen light". Home Assistant calls it
``light.kitchen_ceiling``. This closes that gap, and refuses rather than
guesses when the request is ambiguous --- an assistant that silently picks one
of three lights is worse than one that asks.

Scoring is deliberately simple and inspectable. A confidently wrong match on
something that changes the physical environment is the failure that matters,
so the threshold errs towards asking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.automation.automationProvider import Device

logger = logging.getLogger(__name__)

# Words that carry no identifying information in a spoken request.
STOP_WORDS = frozenset(
    {"the", "a", "an", "my", "our", "please", "in", "on", "at", "to", "of", "room"}
)

# Spoken domain words, so "the kitchen light" prefers a light over a switch.
DOMAIN_WORDS = {
    "light": "light",
    "lights": "light",
    "lamp": "light",
    "lamps": "light",
    "switch": "switch",
    "plug": "switch",
    "socket": "switch",
    "fan": "fan",
    "blind": "cover",
    "blinds": "cover",
    "curtain": "cover",
    "curtains": "cover",
    "thermostat": "climate",
    "heating": "climate",
    "lock": "lock",
    "speaker": "media_player",
}

EXACT_SCORE = 1.0
MINIMUM_SCORE = 0.45
# Two candidates within this of each other are treated as ambiguous.
AMBIGUITY_MARGIN = 0.08


@dataclass(frozen=True, slots=True)
class Match:
    """A device and how well it matched."""

    device: Device
    score: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """What the matcher concluded."""

    best: Device | None
    score: float = 0.0
    alternatives: tuple[Device, ...] = ()
    ambiguous: bool = False

    @property
    def found(self) -> bool:
        return self.best is not None and not self.ambiguous

    def describe(self) -> str:
        if self.ambiguous:
            names = ", ".join(device.describedName for device in self.alternatives)
            return f"that could mean {names}"
        if self.best is None:
            return "no matching device"
        return f"{self.best.describedName} ({self.score:.2f})"


def tokenise(text: str) -> list[str]:
    """Reduce free text to meaningful lowercase words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def domainFor(tokens: list[str]) -> str | None:
    """The device domain a request implies, if any word names one."""
    for token in tokens:
        domain = DOMAIN_WORDS.get(token)
        if domain:
            return domain
    return None


# A domain word implies a kind of device, but not reliably: a smart plug
# driving a fan is a switch in Home Assistant, so "desk fan" must still reach
# it. Mismatches are penalised rather than excluded.
DOMAIN_MISMATCH_PENALTY = 0.6


def scoreDevice(device: Device, tokens: list[str], requestedDomain: str | None) -> float:
    """How well a device answers to the requested words.

    Area and name both count, so that "kitchen light" beats a light merely
    called "Kitchen-style lamp" in another room.
    """
    if not tokens:
        return 0.0

    # A request made only of domain words identifies nothing in particular.
    # "Turn off the light" needs to know which light.
    if all(token in DOMAIN_WORDS for token in tokens):
        return 0.0

    nameTokens = set(tokenise(device.name))
    areaTokens = set(tokenise(device.area or ""))
    idTokens = set(tokenise(device.deviceId.partition(".")[2]))

    matched = 0.0
    for token in tokens:
        if token in nameTokens or token in idTokens or token in areaTokens:
            matched += 1.0
        elif DOMAIN_WORDS.get(token) == device.domain:
            matched += 0.5
        elif any(nameToken.startswith(token) for nameToken in nameTokens | idTokens):
            matched += 0.6

    score = matched / len(tokens)

    # A request naming an area should prefer devices actually in it.
    if areaTokens and areaTokens & set(tokens):
        score += 0.15

    if requestedDomain and device.domain != requestedDomain:
        score *= DOMAIN_MISMATCH_PENALTY

    return min(score, EXACT_SCORE)


def matchDevice(
    query: str,
    devices: list[Device],
    minimumScore: float = MINIMUM_SCORE,
) -> MatchResult:
    """Find the device a spoken request refers to."""
    tokens = tokenise(query)
    if not tokens or not devices:
        return MatchResult(best=None)

    requestedDomain = domainFor(tokens)

    scored = sorted(
        (
            Match(device=device, score=scoreDevice(device, tokens, requestedDomain))
            for device in devices
        ),
        key=lambda match: match.score,
        reverse=True,
    )
    viable = [match for match in scored if match.score >= minimumScore]

    if not viable:
        logger.debug("No device matched %r above %.2f", query, minimumScore)
        return MatchResult(best=None, alternatives=tuple(match.device for match in scored[:3]))

    best = viable[0]
    close = [match for match in viable[1:] if best.score - match.score <= AMBIGUITY_MARGIN]

    if close:
        logger.info(
            "Request %r is ambiguous between %d devices", query, len(close) + 1
        )
        return MatchResult(
            best=best.device,
            score=best.score,
            alternatives=tuple([best.device, *(match.device for match in close)]),
            ambiguous=True,
        )

    return MatchResult(
        best=best.device,
        score=best.score,
        alternatives=tuple(match.device for match in viable[1:4]),
    )


def matchDevices(
    query: str,
    devices: list[Device],
    minimumScore: float = MINIMUM_SCORE,
) -> list[Device]:
    """Every device matching a request, for actions that address a group.

    "Turn off the kitchen lights" should reach all of them, so this returns the
    full viable set rather than resolving to one.
    """
    tokens = tokenise(query)
    if not tokens or not devices:
        return []

    requestedDomain = domainFor(tokens)
    matches = [
        Match(device=device, score=scoreDevice(device, tokens, requestedDomain))
        for device in devices
    ]
    return [
        match.device
        for match in sorted(matches, key=lambda match: match.score, reverse=True)
        if match.score >= minimumScore
    ]
