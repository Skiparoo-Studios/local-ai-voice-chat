"""The voice-recording checker.

Its noise measurement was wrong twice before it was right, in both directions:
first reporting clean speech as noisy, then passing audibly hissy speech as
clean. The threshold is calibrated against the committed utterance, so this
pins it rather than trusting it.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from app.audio.audioBuffer import AudioBuffer

FIXTURE = Path(__file__).parents[1] / "data" / "utterance.wav"
SCRIPT = Path(__file__).parents[2] / "scripts" / "checkVoice.py"


def loadChecker():
    """Import the script, which is not part of the package.

    Registered in ``sys.modules`` before it runs, because a dataclass using
    slots looks its own module up while the class is being built.
    """
    specification = importlib.util.spec_from_file_location("checkVoice", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    sys.modules["checkVoice"] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    if not SCRIPT.is_file():
        pytest.skip("checkVoice.py is not present")
    return loadChecker()


@pytest.fixture(scope="module")
def speech() -> list[float]:
    if not FIXTURE.is_file():
        pytest.skip(f"missing fixture {FIXTURE}")
    # Repeated so there is enough material to measure.
    return AudioBuffer.fromWavFile(FIXTURE).toFloatSamples() * 3


def write(path: Path, samples: list[float], rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioBuffer.fromFloatSamples(samples, rate).writeWav(path)
    return path


class TestNoiseDetection:
    def testCleanSpeechPasses(self, checker, speech, tmp_path: Path):
        report = checker.analyse(write(tmp_path / "clean.wav", speech))

        assert report.problems == []
        assert report.signalToNoise > checker.POOR_SIGNAL_TO_NOISE

    def testAudibleHissIsCaught(self, checker, speech, tmp_path: Path):
        """The failure that got through: noise that fills the pauses."""
        random.seed(0)
        hissy = [
            max(-1.0, min(1.0, value + random.uniform(-0.02, 0.02))) for value in speech
        ]

        report = checker.analyse(write(tmp_path / "hiss.wav", hissy))

        assert report.signalToNoise < checker.POOR_SIGNAL_TO_NOISE
        assert any("background noise" in problem for problem in report.problems)

    def testCleanAndHissyAreClearlyApart(self, checker, speech, tmp_path: Path):
        """They measured 16 dB and 15 dB before the percentile was corrected."""
        random.seed(0)
        hissy = [
            max(-1.0, min(1.0, value + random.uniform(-0.02, 0.02))) for value in speech
        ]

        clean = checker.analyse(write(tmp_path / "a.wav", speech))
        noisy = checker.analyse(write(tmp_path / "b.wav", hissy))

        assert clean.signalToNoise - noisy.signalToNoise > 20


class TestLevelDetection:
    def testQuietRecordingIsCaught(self, checker, speech, tmp_path: Path):
        faint = [value * 0.05 for value in speech]

        report = checker.analyse(write(tmp_path / "faint.wav", faint))

        assert any("quiet" in problem for problem in report.problems)

    def testClippedRecordingIsCaught(self, checker, speech, tmp_path: Path):
        loud = [max(-1.0, min(1.0, value * 4)) for value in speech]

        report = checker.analyse(write(tmp_path / "loud.wav", loud))

        assert any("clipped" in problem for problem in report.problems)

    def testLowSampleRateIsCaught(self, checker, speech, tmp_path: Path):
        report = checker.analyse(write(tmp_path / "low.wav", speech[:16000], rate=8000))

        assert any("22050" in problem for problem in report.problems)


class TestMeasurement:
    def testSpeechAndSilenceAreSeparated(self, checker, speech, tmp_path: Path):
        padded = [0.0] * 24000 + speech + [0.0] * 24000

        report = checker.analyse(write(tmp_path / "padded.wav", padded))

        assert report.speechSeconds < report.seconds
        assert report.silenceShare > 0.15

    def testPercentileHandlesAnEmptyList(self, checker):
        assert checker.percentile([], 0.5) == 0.0
