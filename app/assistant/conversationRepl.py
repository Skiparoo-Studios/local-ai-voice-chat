"""Interactive conversation prompt.

The Stage 3 milestone: press Enter, speak, and hear a spoken reply. Typed input
runs the same turn without the microphone, which is how the loop is exercised
in tests and on machines with no working capture device.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.assistant.runtime import AssistantRuntime, TurnResult
from app.assistant.toddlerCommands import runToddlerCommand
from app.audio.input import AudioInputError
from app.events import AssistantState
from app.speech.repl import cleanInputLine
from app.speech.speechToText import TranscriptionError
from app.speech.textToSpeech import SpeechSynthesisError

logger = logging.getLogger(__name__)

HELP_TEXT = """
Press Enter to speak, or type a message to send it without the microphone.

  :help            show this message
  :history         show the conversation so far
  :clear           forget the conversation
  :models          list models the language model service offers
  :toddler         toddler mode and its questions (:toddler alone to see them)
  :info            show providers, session and state
  :quit            exit (Ctrl-C and Ctrl-D also work)
""".strip()


class ConversationRepl:
    """Drives conversation turns from the terminal."""

    def __init__(
        self,
        runtime: AssistantRuntime,
        *,
        outputsDirectory: Path,
        useMicrophone: bool = True,
    ) -> None:
        self._runtime = runtime
        self._outputsDirectory = outputsDirectory
        self._useMicrophone = useMicrophone
        self._running = False

    async def run(self) -> int:
        self._running = True
        print(HELP_TEXT)
        if self._useMicrophone:
            # This mode waits for a keypress by design. Hands-free listening is
            # a different mode, and the wake-word settings belong to that one,
            # so someone who has configured them here needs telling.
            print("\nFor hands-free listening, with no keypress: --mode wake")
        print()

        while self._running:
            prompt = "[Enter to speak] > " if self._useMicrophone else "> "
            try:
                line = await asyncio.to_thread(input, prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = cleanInputLine(line)

            try:
                if line.startswith(":"):
                    await self._handleCommand(line)
                elif line:
                    self._report(await self._runtime.handleText(line))
                elif self._useMicrophone:
                    self._report(await self._spokenTurn())
                else:
                    print("  type a message; :help for commands")
            except KeyboardInterrupt:
                print("\n(interrupted)")
                await self._runtime.interrupt()
            except (TranscriptionError, SpeechSynthesisError, AudioInputError) as error:
                print(f"error: {error}")

        return 0

    # --- Turns -----------------------------------------------------------

    async def _spokenTurn(self) -> TurnResult:
        """Record until Enter is pressed, then run the turn."""
        stop = asyncio.Event()
        turn = asyncio.create_task(self._runtime.runTurn(stopWhen=stop))

        waiting = asyncio.create_task(
            asyncio.to_thread(input, "  listening... [Enter when done] ")
        )

        # The turn continues past the end of capture, so wait for whichever
        # comes first and then release the other.
        done, _ = await asyncio.wait(
            {turn, waiting}, return_when=asyncio.FIRST_COMPLETED
        )

        stop.set()
        if waiting in done:
            result = await turn
        else:
            print("  (reached the maximum recording length)")
            result = await turn
            waiting.cancel()

        return result

    def _report(self, result: TurnResult) -> None:
        if result.userText:
            print(f"  you:       {result.userText}")
        if result.assistantText:
            print(f"  assistant: {result.assistantText}")
        elif not result.userText:
            print("  (nothing was said)")

        if result.handled:
            print(f"  {result.timing.describe()}")

        if result.shouldStop:
            self._running = False

    # --- Commands --------------------------------------------------------

    async def _handleCommand(self, line: str) -> None:
        command, _, argument = line[1:].partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in {"quit", "exit", "q"}:
            self._running = False
        elif command in {"help", "h", "?"}:
            print(HELP_TEXT)
        elif command == "history":
            self._printHistory()
        elif command == "clear":
            self._runtime.conversation.clear()
            print("  conversation cleared")
        elif command == "info":
            self._printInfo()
        elif command == "models":
            await self._printModels()
        elif command == "toddler":
            for message in runToddlerCommand(self._runtime.handler, argument):
                print(message)
        else:
            print(f"unknown command {command!r}; try :help")

    async def _printModels(self) -> None:
        """List what the configured language model service offers."""
        provider = getattr(self._runtime.handler, "provider", None)
        if provider is None:
            print("  the current handler does not use a language model")
            return

        if not await provider.isAvailable():
            print(f"  {provider.describe()} is not reachable")
            return

        models = await provider.listModels()
        print(f"  {', '.join(models) if models else 'the service did not list any models'}")

    def _printHistory(self) -> None:
        conversation = self._runtime.conversation
        if not len(conversation):
            print("  nothing said yet")
            return
        for turn in conversation.turns:
            speaker = "you" if turn.role.value == "user" else "assistant"
            print(f"  {speaker:>9}: {turn.text}")
        print(f"  ({conversation.describe()})")

    def _printInfo(self) -> None:
        print(f"  {self._runtime.describe()}")
        print(f"  session:  {self._runtime.session.describe()}")
        print(f"  state:    {self._runtime.state.value}")
        print(f"  input:    {'microphone' if self._useMicrophone else 'typed text only'}")


def describeStateSequence(states: list[AssistantState]) -> str:
    """Render an observed state sequence, for logging and tests."""
    return " -> ".join(state.value for state in states)
