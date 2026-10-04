"""Resolving spoken device names to entity identifiers."""

from __future__ import annotations

import pytest

from app.automation.automationProvider import Device
from app.automation.deviceMatcher import matchDevice, matchDevices, tokenise

DEVICES = [
    Device("light.kitchen_ceiling", "Ceiling light", "light", "Kitchen", "off"),
    Device("light.kitchen_under_cabinet", "Under cabinet light", "light", "Kitchen", "off"),
    Device("light.lounge_lamp", "Lamp", "light", "Lounge room", "on"),
    Device("light.bedroom_ceiling", "Ceiling light", "light", "Bedroom", "off"),
    Device("switch.coffee_machine", "Coffee machine", "switch", "Kitchen", "off"),
    Device("switch.desk_fan", "Desk fan", "switch", "Study", "off"),
    Device("climate.lounge", "Thermostat", "climate", "Lounge room", "21"),
]


class TestTokenising:
    def testStopWordsAreDropped(self):
        assert tokenise("the kitchen light") == ["kitchen", "light"]

    def testPunctuationAndCaseAreIgnored(self):
        assert tokenise("Kitchen Light!") == ["kitchen", "light"]

    def testRoomIsTreatedAsNoise(self):
        assert tokenise("the lounge room lamp") == ["lounge", "lamp"]


class TestMatching:
    def testAreaAndDomainTogetherResolve(self):
        result = matchDevice("bedroom light", DEVICES)

        assert result.found
        assert result.best.deviceId == "light.bedroom_ceiling"

    def testDeviceNameResolves(self):
        result = matchDevice("coffee machine", DEVICES)

        assert result.found
        assert result.best.deviceId == "switch.coffee_machine"

    def testDomainWordSelectsTheRightKind(self):
        """'lounge lamp' must find the light, not the thermostat in that room."""
        result = matchDevice("lounge lamp", DEVICES)

        assert result.found
        assert result.best.domain == "light"

    def testThermostatIsFoundByItsSpokenWord(self):
        result = matchDevice("lounge thermostat", DEVICES)

        assert result.found
        assert result.best.deviceId == "climate.lounge"

    def testUnknownDeviceIsNotMatched(self):
        assert not matchDevice("garage door", DEVICES).found

    def testEmptyQueryMatchesNothing(self):
        assert not matchDevice("", DEVICES).found

    def testEmptyDeviceListMatchesNothing(self):
        assert not matchDevice("kitchen light", []).found


class TestAmbiguity:
    """Silently picking one of several devices is the failure that matters."""

    def testTwoEquallyGoodMatchesAreAmbiguous(self):
        result = matchDevice("kitchen light", DEVICES)

        assert result.ambiguous
        assert not result.found
        assert len(result.alternatives) == 2

    def testAmbiguityIsDescribedForSpeaking(self):
        result = matchDevice("kitchen light", DEVICES)

        assert "could mean" in result.describe()

    def testAUniqueMatchIsNotAmbiguous(self):
        assert not matchDevice("under cabinet light", DEVICES).ambiguous


class TestGroupMatching:
    def testAllKitchenLightsAreReturned(self):
        matched = matchDevices("kitchen lights", DEVICES)

        assert {device.deviceId for device in matched} == {
            "light.kitchen_ceiling",
            "light.kitchen_under_cabinet",
        }

    def testGroupMatchRespectsDomain(self):
        """The coffee machine is in the kitchen but is not a light."""
        matched = matchDevices("kitchen lights", DEVICES)

        assert all(device.domain == "light" for device in matched)

    def testNoMatchesReturnsEmpty(self):
        assert matchDevices("garage", DEVICES) == []


class TestScoring:
    def testDomainWordAloneDoesNotMatchEverything(self):
        """'light' on its own names no particular device."""
        result = matchDevice("light", DEVICES)

        assert not result.found

    def testMoreSpecificQueryScoresHigher(self):
        vague = matchDevice("bedroom light", DEVICES)
        specific = matchDevice("under cabinet light", DEVICES)

        assert specific.score >= vague.score - 0.2
        assert specific.found

    def testScoreIsBounded(self):
        result = matchDevice("kitchen coffee machine", DEVICES)

        assert 0.0 <= result.score <= 1.0

    def testRaisingTheThresholdRejectsWeakMatches(self):
        assert not matchDevice("lamp", DEVICES, minimumScore=0.99).found


class TestPartialWords:
    def testPrefixMatchesArePermitted(self):
        result = matchDevice("study fan", DEVICES)

        assert result.found
        assert result.best.deviceId == "switch.desk_fan"

    def testSpokenDomainNeedNotMatchTheEntityDomain(self):
        """A smart plug driving a fan is a switch, but is still 'the fan'."""
        result = matchDevice("desk fan", DEVICES)

        assert result.found
        assert result.best.deviceId == "switch.desk_fan"
        assert result.best.domain == "switch"

    def testDomainStillBreaksTiesWhenItMatches(self):
        """'lounge lamp' must find the light, not the thermostat in that room."""
        result = matchDevice("lounge lamp", DEVICES)

        assert result.best.domain == "light"

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("under cabinet light", "light.kitchen_under_cabinet"),
            ("desk fan", "switch.desk_fan"),
            ("hallway", None),
        ],
    )
    def testAssortedQueries(self, query: str, expected: str | None):
        result = matchDevice(query, DEVICES)

        assert (result.best.deviceId if result.found else None) == expected
