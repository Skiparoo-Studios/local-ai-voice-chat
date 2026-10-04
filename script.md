# Voice sample script

Read this aloud to record a voice for the assistant. Three takes, about fifteen
seconds each, saved as `voices/<yourname>/one.wav`, `two.wav`, `three.wav`.

Before you start:

- **Say it, don't read it.** XTTS copies your delivery as faithfully as your
  voice. Talk as though to someone in the next room. A passage *read* produces
  an assistant that reads at you, which is the usual disappointment.
- **Quiet room.** No fan, no traffic, no music. Room tone gets cloned too, and
  ends up underneath every reply.
- **Don't touch the recording level between takes.** Loudness is not
  normalised, so takes at different levels blend badly.
- **Keep the breaths in.** Their absence is audible.

A hand-span from the microphone, slightly off to one side so plosives do not
thump. Then check it with `python scripts\checkVoice.py <yourname>`.

---

## Take one

> Good morning. The kitchen is warm and the kettle has just boiled.
>
> I have turned the hallway lights down to about half brightness, and the back
> door is locked. Nothing else needs your attention.

---

## Take two

> Would you like the heating on before you get home?
>
> There are three things on the calendar: a dentist at nine thirty, lunch with
> Joanne, and a parcel arriving some time after four.
>
> Shall I read them again?

---

## Take three

> The temperature outside is six degrees, falling to minus one overnight.
>
> The wireless password is J, X, seven, four, Q, zero.
>
> Just think, a huge shipment of vexed zebras arrived by ship in Oslo.

---

The last sentence is there on purpose. It carries the consonants ordinary
sentences skip --- the *zh* in *huge*, along with x, v, z, sh and th --- in
something you can still say without sounding daft.

## Afterwards

```powershell
python scripts\checkVoice.py michael
python -m app.main --voice michael --say "The kitchen light is now off."
```

Set the voice permanently in `config/settings.json`:

```json
{ "textToSpeech": { "voice": "michael" } }
```

`docs/recordingAVoice.md` covers how much of the recording is used, what the
checker measures, and what to do when a clone comes out wrong.
