"""Check a voice recording before asking XTTS to clone it.

Most disappointing clones come from the recording, not the model, and the
faults are measurable: too short, too quiet, clipped, mostly silence, or a
noise floor that ends up under every reply. This reports them.

Usage:
    python scripts/checkVoice.py michael
    python scripts/checkVoice.py --list
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from app.audio.audioBuffer import AudioBuffer
from app.audio.input import resampleBuffer
from app.speech.voices import VoiceLibrary, VoiceNotFoundError

# XTTS uses up to thirty seconds; beyond that nothing more is read.
USEFUL_SECONDS = 30.0
# Below this there is too little to characterise a voice.
MINIMUM_SECONDS = 6.0

# Readable, but throwing away detail the model would otherwise clone.
LOSSY_FORMATS = frozenset({".mp3", ".ogg", ".m4a"})

# Peak level. Above this, clipping is likely; below it, the recording is quiet
# enough that the noise floor starts to matter.
CLIPPING_PEAK = 0.99
QUIET_PEAK = 0.15

# A frame this far below the loudest part is silence rather than speech.
SILENCE_BELOW_DECIBELS = 40.0
FRAME_SECONDS = 0.03

# The noise floor is read from the quietest twentieth of the recording, which
# in speech falls in the pauses between phrases. Calibrated rather than
# guessed: at the tenth percentile, clean speech and audibly hissy speech both
# measure about 16 dB, because that percentile lands in quiet phonemes rather
# than silence. At the twentieth they separate cleanly, 48 dB against 20 dB.
NOISE_PERCENTILE = 0.05
# Some silence has to exist for any of this to mean anything.
MINIMUM_SILENCE_SHARE = 0.05
# Below this, background noise is audible under a synthesised reply. Clean
# speech measures far higher; the hissy recording used to calibrate this
# measured 20 dB.
POOR_SIGNAL_TO_NOISE = 30.0


@dataclass(slots=True)
class Report:
    """What was measured about one file."""

    path: Path
    seconds: float
    sampleRate: int
    channels: int
    peak: float
    speechSeconds: float
    noiseFloorDecibels: float
    signalToNoise: float
    problems: list[str]
    notes: list[str]

    @property
    def silenceShare(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return 1.0 - (self.speechSeconds / self.seconds)


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a voice recording")
    parser.add_argument("voice", nargs="?", help="the voice to check")
    parser.add_argument(
        "--list", action="store_true", dest="listVoices", help="list known voices"
    )
    parser.add_argument("--voices", type=Path, default=Path("voices"))
    return parser.parse_args()


def decibels(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else -120.0


class UnreadableAudioError(Exception):
    """Raised when a recording cannot be decoded for measurement."""


def readAudio(path: Path) -> tuple[list[float], int, int]:
    """Decode any format the voice library accepts, as mono float samples.

    Recordings arrive as whatever the recorder produced, often MP3, and a
    checker that only understands WAV can say nothing about the files people
    actually have.
    """
    if path.suffix.lower() == ".wav":
        buffer = AudioBuffer.fromWavFile(path)
        mono = resampleBuffer(buffer, buffer.sampleRate)
        return mono.toFloatSamples(), buffer.sampleRate, buffer.channels

    try:
        import librosa
    except ImportError as error:
        raise UnreadableAudioError(
            f"{path.suffix} needs librosa to measure; "
            'install with: pip install -e ".[tts]"'
        ) from error

    try:
        samples, rate = librosa.load(str(path), sr=None, mono=False)
    except Exception as error:
        raise UnreadableAudioError(f"could not decode this file: {error}") from error

    if getattr(samples, "ndim", 1) == 2:
        channels = int(samples.shape[0])
        samples = samples.mean(axis=0)
    else:
        channels = 1

    return [float(value) for value in samples], int(rate), channels


def analyse(path: Path) -> Report:
    try:
        samples, sampleRate, channels = readAudio(path)
    except UnreadableAudioError as error:
        return Report(
            path=path,
            seconds=0.0,
            sampleRate=0,
            channels=0,
            peak=0.0,
            speechSeconds=0.0,
            noiseFloorDecibels=0.0,
            signalToNoise=0.0,
            problems=[str(error)],
            notes=[],
        )

    mono = AudioBuffer.fromFloatSamples(samples, sampleRate)
    peak = max((abs(sample) for sample in samples), default=0.0)

    # Split into short frames and call the quiet ones silence, so that a long
    # pause at the start or end is visible rather than counted as speech.
    frameLength = max(1, int(mono.sampleRate * FRAME_SECONDS))
    loudest = 0.0
    frames: list[float] = []
    for start in range(0, len(samples) - frameLength + 1, frameLength):
        window = samples[start : start + frameLength]
        level = math.sqrt(sum(value * value for value in window) / len(window))
        frames.append(level)
        loudest = max(loudest, level)

    threshold = loudest * (10.0 ** (-SILENCE_BELOW_DECIBELS / 20.0))
    speaking = [level for level in frames if level > threshold]
    speechSeconds = len(speaking) * FRAME_SECONDS

    ordered = sorted(frames)
    speechLevel = median(sorted(speaking)) if speaking else 0.0
    noiseFloor = decibels(percentile(ordered, NOISE_PERCENTILE))
    signalToNoise = decibels(speechLevel) - noiseFloor if speechLevel else 0.0

    # A recording whose quietest moments are still close to its loudest either
    # has noise filling the pauses or has no pauses at all. Both are worth
    # looking at, and they are not distinguishable from here.
    pauses = 1.0 - (len(speaking) / len(frames)) if frames else 0.0

    problems: list[str] = []
    notes: list[str] = []

    if peak >= CLIPPING_PEAK:
        problems.append("clipped: the loudest parts hit the ceiling and are distorted")
    elif peak < QUIET_PEAK:
        problems.append(
            f"quiet: peaks at {decibels(peak):.0f} dB, so the noise floor is close behind"
        )

    if channels > 1:
        notes.append(f"{channels} channels, which will be mixed down to mono")
    if sampleRate < 22050:
        problems.append(
            f"{sampleRate} Hz is below the 22050 Hz XTTS reads at, so detail is missing"
        )
    if path.suffix.lower() in LOSSY_FORMATS:
        notes.append(
            f"{path.suffix} is lossy; uncompressed audio clones a little more "
            "faithfully, though a good recording matters far more than the format"
        )

    silenceShare = 1.0 - (speechSeconds / mono.durationSeconds) if mono.durationSeconds else 0
    if silenceShare > 0.4:
        problems.append(
            f"{silenceShare * 100:.0f}% is silence, so there is less voice here than it looks"
        )

    if speechLevel and signalToNoise < POOR_SIGNAL_TO_NOISE:
        cause = (
            "background noise filling the pauses"
            if pauses < MINIMUM_SILENCE_SHARE
            else "background noise"
        )
        problems.append(
            f"only {signalToNoise:.0f} dB between the speech and the quietest "
            f"part, which suggests {cause}. Listen to a pause and check it is "
            "actually silent."
        )

    return Report(
        path=path,
        seconds=mono.durationSeconds,
        sampleRate=sampleRate,
        channels=channels,
        peak=peak,
        speechSeconds=speechSeconds,
        noiseFloorDecibels=noiseFloor,
        signalToNoise=signalToNoise,
        problems=problems,
        notes=notes,
    )


def percentile(ordered: list[float], fraction: float) -> float:
    """A value from an already-sorted list, by position."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def median(ordered: list[float]) -> float:
    return percentile(ordered, 0.5)


def describe(report: Report) -> None:
    print(f"\n  {report.path.name}")
    if report.sampleRate:
        background = (
            f"{report.signalToNoise:.0f} dB above the background"
            if report.signalToNoise
            else "background not measurable"
        )
        print(
            f"    {report.seconds:.1f}s, {report.speechSeconds:.1f}s of speech, "
            f"{report.sampleRate} Hz, peak {decibels(report.peak):.0f} dB, {background}"
        )
    for note in report.notes:
        print(f"    note: {note}")
    for problem in report.problems:
        print(f"    PROBLEM: {problem}")
    if report.sampleRate and not report.problems:
        print("    looks good")


def main() -> int:
    arguments = parseArguments()
    library = VoiceLibrary(arguments.voices)

    if arguments.listVoices or not arguments.voice:
        names = library.listVoices()
        if not names:
            print(f"No voices in {arguments.voices}.")
            print("See docs/recordingAVoice.md for how to make one.")
            return 1
        print("Voices:")
        for name in names:
            print(f"  {name}")
        return 0

    try:
        voice = library.get(arguments.voice)
    except VoiceNotFoundError as error:
        print(error)
        return 1

    print(f"{voice.name}: {voice.sampleCount} file(s) in {arguments.voices / voice.name}")

    reports = [analyse(path) for path in voice.samples]
    for report in reports:
        describe(report)

    total = sum(report.seconds for report in reports)
    speech = sum(report.speechSeconds for report in reports)
    used = min(total, USEFUL_SECONDS)

    print(f"\n  {total:.1f}s in total, {speech:.1f}s of it speech")
    print(f"  XTTS will use the first {used:.1f}s")

    if total > USEFUL_SECONDS:
        print(
            f"  the last {total - USEFUL_SECONDS:.1f}s is beyond what XTTS reads, "
            "which is harmless but does nothing"
        )

    problems = [problem for report in reports for problem in report.problems]

    if speech < MINIMUM_SECONDS:
        print(
            f"\n  TOO SHORT: {speech:.1f}s of speech is below the {MINIMUM_SECONDS:.0f}s "
            "needed to characterise a voice. Record more."
        )
        return 1

    if problems:
        print(f"\n  {len(problems)} thing(s) worth fixing before cloning. See above.")
        return 1

    print("\n  Ready to clone. Hear it with:")
    print(f'    python -m app.main --voice {voice.name} --say "The kitchen light is now off."')
    return 0


if __name__ == "__main__":
    sys.exit(main())
