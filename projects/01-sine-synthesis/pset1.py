#####################################################################
#
# This software is to be used for MIT's class Interactive Music Systems only.
# Since this file may contain answers to homework problems, you MAY NOT release it publicly.
#
#####################################################################


import sys
import os


sys.path.insert(0, os.path.abspath(".."))


from imslib.core import BaseWidget, run
from imslib.audio import Audio
from imslib.writer import AudioWriter
from imslib.gfxutil import topleft_label


import numpy as np


# shared helpers (used by tests and NoteGenerator)
def midi_to_hz(pitch):
    return 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))


def harmonic_amplitudes(wave, num_harmonics=10):
    assert wave in ["sine", "square", "triangle", "sawtooth"]
    n = np.arange(1, int(num_harmonics) + 1, dtype=np.float64)

    if wave == "sine":
        amps = np.zeros_like(n)
        amps[0] = 1.0
        return amps

    if wave == "square":
        return np.where((n % 2) == 1, 1.0 / n, 0.0)

    if wave == "sawtooth":
        return 1.0 / n

    # triangle: odd harmonics only, alternating sign, 1/n^2 rolloff
    odd = ((n % 2) == 1)
    signs = (-1.0) ** ((n - 1.0) / 2.0)
    return np.where(odd, signs * (1.0 / (n * n)), 0.0)


def instrument_harmonics(name, num_harmonics=10):
    """
    Approximate woodwind spectral envelopes using low-order harmonic models.

    These are compact perceptual models (not physical simulations):
    - flute: fundamental-heavy and soft high-frequency rolloff
    - clarinet: odd-harmonic dominant closed-pipe behavior
    - oboe: stronger upper partial content (bright reed tone)
    - tenor sax: rich spectrum with moderate brightness
    - bassoon: darker low-mid emphasis with faster high rolloff
    """
    n = np.arange(1, int(num_harmonics) + 1, dtype=np.float64)

    if name == "flute":
        amps = 1.0 / (n ** 2.35)
        amps[1] *= 0.55  # keep 2nd harmonic present but soft
        amps[2] *= 0.75
    elif name == "clarinet":
        odd = (n % 2) == 1
        amps = np.where(odd, 1.0 / (n ** 1.05), 0.12 / (n ** 1.4))
    elif name == "oboe":
        amps = 1.0 / (n ** 0.95)
        amps *= np.where(n <= 6, 1.25, 0.9)  # emphasize low-mid overtones
    elif name == "tenor_sax":
        odd = (n % 2) == 1
        amps = 1.0 / (n ** 0.85)
        amps *= np.where(odd, 1.12, 0.92)  # gentle odd preference
    elif name == "bassoon":
        amps = 1.0 / (n ** 1.6)
        amps *= np.where(n <= 4, 1.15, 0.7)
    else:
        raise ValueError(f"unsupported instrument harmonic profile: {name}")

    # Keep relative timbre while reducing clipping risk from harmonic summation.
    scale = float(np.sum(np.abs(amps)))
    if scale > 0.0:
        amps = amps / scale
    return amps




# part 1
class NoteGenerator(object):
    A4_PITCH = 69
    A4_FREQ = 440.0

    def __init__(self, pitch, gain, wave="sine", sample_rate=44100):
        super(NoteGenerator, self).__init__()
        assert wave in ["sine", "square", "triangle", "sawtooth"]  # for part 4

        self.pitch = float(pitch)
        self.gain = float(gain)
        self.wave = wave

        # audio state
        self.sample_rate = float(sample_rate)
        self.phase = 0.0
        self.is_on = True  # becomes False after note_off()
        self.num_harmonics = 10

        # precompute frequency and phase increment
        self.freq = self.midi_to_hz(self.pitch)
        self.phase_inc = (2.0 * np.pi * self.freq) / self.sample_rate
        self.harmonics = harmonic_amplitudes(self.wave, self.num_harmonics)

    @classmethod
    def midi_to_hz(cls, pitch: float) -> float:
        return midi_to_hz(pitch)

    def note_off(self):
        # Next generate() should report done
        self.is_on = False

    def generate(self, num_frames, num_channels):
        assert num_channels == 1

        if not self.is_on:
            # Return anything (usually zeros) and False to indicate finished
            return (np.zeros(num_frames, dtype=np.float32), False)

        # Generate waveform chunk from harmonic recipe.
        # phases: phase, phase+inc, ..., phase+(num_frames-1)*inc
        phases = self.phase + self.phase_inc * np.arange(num_frames, dtype=np.float64)
        output = np.zeros(num_frames, dtype=np.float64)
        for k, amp in enumerate(self.harmonics, start=1):
            if amp != 0.0:
                output += amp * np.sin(k * phases)
        output = (self.gain * output).astype(np.float32)

        # advance phase (wrap to avoid floating-point blowup over time)
        self.phase = float((self.phase + self.phase_inc * num_frames) % (2.0 * np.pi))

        return (output, True)



# part 2
# Generate the envelope with this class, as described in Musimathics Vol 2, 9.2.1
# Use decay_time and n2 to modify the envelope
# BONUS: use attack_time and n1 to modify the envelope
class Envelope(object):
    def __init__(self, input_generator, attack_time=0.0, n1=1.0, decay_time=0.0, n2=1.0):
        super(Envelope, self).__init__()
        self.input_generator = input_generator
        self.sample_rate = float(getattr(input_generator, "sample_rate", 44100))

        self.attack_time = float(max(attack_time, 0.0))
        self.decay_time = float(max(decay_time, 0.0))
        self.n1 = float(max(n1, 0.0))
        self.n2 = float(max(n2, 0.0))

        self.attack_samps = int(round(self.attack_time * self.sample_rate))
        self.decay_samps = int(round(self.decay_time * self.sample_rate))
        self.total_samps = self.attack_samps + self.decay_samps
        self.pos = 0


    def generate(self, num_frames, num_channels):
        assert num_channels == 1

        if self.pos >= self.total_samps:
            return (np.zeros(num_frames, dtype=np.float32), False)

        source, source_continue = self.input_generator.generate(num_frames, num_channels)
        source = np.asarray(source, dtype=np.float32)

        idx = np.arange(self.pos, self.pos + num_frames, dtype=np.float64)
        env = np.zeros(num_frames, dtype=np.float64)

        if self.attack_samps > 0:
            attack_mask = idx < self.attack_samps
            attack_idx = idx[attack_mask]
            env[attack_mask] = (attack_idx / float(self.attack_samps)) ** self.n1
        else:
            attack_mask = np.zeros(num_frames, dtype=bool)

        decay_mask = ~attack_mask
        if self.decay_samps > 0:
            decay_idx = idx[decay_mask] - self.attack_samps
            decay_phase = 1.0 - (decay_idx / float(self.decay_samps))
            env[decay_mask] = np.maximum(decay_phase, 0.0) ** self.n2

        output = source * env.astype(np.float32)
        self.pos += num_frames
        continue_flag = (self.pos < self.total_samps) and source_continue
        return (output, continue_flag)




# part 3
class Mixer(object):
    def __init__(self):
        super(Mixer, self).__init__()
        self.generators = []
        self.gain = 0.1


    def add(self, gen):
        self.generators.append(gen)


    def remove(self, gen):
        if gen in self.generators:
            self.generators.remove(gen)


    def get_num_generators(self):
        return len(self.generators)


    def set_gain(self, gain):
        self.gain = float(np.clip(gain, 0.0, 1.0))


    def generate(self, num_frames, num_channels):
        output = np.zeros(num_frames * num_channels, dtype=np.float32)
        done = []

        for gen in self.generators:
            gen_output, keep_going = gen.generate(num_frames, num_channels)
            output += np.asarray(gen_output, dtype=np.float32)
            if not keep_going:
                done.append(gen)

        for gen in done:
            self.remove(gen)

        output *= self.gain
        return (output, True)


class MultiWaveNoteGenerator(object):
    """Generate one note by summing a supplied harmonic recipe."""

    def __init__(
        self,
        pitch,
        gain,
        wave_weights=None,
        harmonic_profile=None,
        sample_rate=44100,
        num_harmonics=10,
        vibrato_rate_hz=0.0,
        vibrato_depth_semitones=0.0,
    ):
        super(MultiWaveNoteGenerator, self).__init__()
        self.pitch = float(pitch)
        self.gain = float(gain)
        self.sample_rate = float(sample_rate)
        self.num_harmonics = int(num_harmonics)
        self.phase = 0.0
        self.is_on = True
        self.frame = 0
        self.vibrato_rate_hz = float(max(vibrato_rate_hz, 0.0))
        self.vibrato_depth_semitones = float(max(vibrato_depth_semitones, 0.0))

        self.freq = midi_to_hz(self.pitch)
        self.phase_inc = (2.0 * np.pi * self.freq) / self.sample_rate

        if harmonic_profile is not None:
            self.harmonics = np.asarray(harmonic_profile, dtype=np.float64)
        else:
            self.harmonics = np.zeros(self.num_harmonics, dtype=np.float64)
            weight_sum = 0.0
            wave_weights = wave_weights or {}
            for wave, weight in wave_weights.items():
                if weight <= 0.0:
                    continue
                self.harmonics += float(weight) * harmonic_amplitudes(wave, self.num_harmonics)
                weight_sum += float(weight)
            if weight_sum > 0:
                self.harmonics /= weight_sum

    def note_off(self):
        self.is_on = False

    def generate(self, num_frames, num_channels):
        assert num_channels == 1
        if not self.is_on:
            return (np.zeros(num_frames, dtype=np.float32), False)

        if self.vibrato_rate_hz > 0.0 and self.vibrato_depth_semitones > 0.0:
            # Frequency-domain vibrato: modulate pitch in semitones over time.
            frames = np.arange(self.frame, self.frame + num_frames, dtype=np.float64)
            vib = np.sin((2.0 * np.pi * self.vibrato_rate_hz * frames) / self.sample_rate)
            inst_freq = self.freq * (2.0 ** ((self.vibrato_depth_semitones * vib) / 12.0))
            phase_incs = (2.0 * np.pi * inst_freq) / self.sample_rate
            phases = self.phase + np.cumsum(phase_incs) - phase_incs[0]
            self.phase = float((phases[-1] + phase_incs[-1]) % (2.0 * np.pi))
            self.frame += num_frames
        else:
            phases = self.phase + self.phase_inc * np.arange(num_frames, dtype=np.float64)
            self.phase = float((self.phase + self.phase_inc * num_frames) % (2.0 * np.pi))

        output = np.zeros(num_frames, dtype=np.float64)
        for k, amp in enumerate(self.harmonics, start=1):
            if amp != 0.0:
                output += amp * np.sin(k * phases)
        output = (self.gain * output).astype(np.float32)
        return (output, True)


class OnePoleLowpass(object):
    """Simple one-pole low-pass filter used as a timbre shaper."""

    def __init__(self, input_generator, alpha=0.2):
        super(OnePoleLowpass, self).__init__()
        self.input_generator = input_generator
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.prev = 0.0

    def generate(self, num_frames, num_channels):
        assert num_channels == 1
        x, continue_flag = self.input_generator.generate(num_frames, num_channels)
        x = np.asarray(x, dtype=np.float32)
        y = np.empty_like(x)
        a = self.alpha
        p = self.prev

        for i in range(num_frames):
            p = p + a * (float(x[i]) - p)
            y[i] = p

        self.prev = p
        return (y, continue_flag)




# part 4: You might find it helpful to create a separate function that
# does the work of adding all the overtones together, and then returning
# that array back to NoteGenerator.




# parts 1-4: Add code to this class as needed to test parts 1-4.
# After finishing part 4, make sure that parts 1-3 still run and are testable.
class MainWidget1(BaseWidget):
    def __init__(self):
        super(MainWidget1, self).__init__()


        self.audio = Audio(1)


        # for debugging audio
        self.writer = AudioWriter("data")
        self.audio.add_listen_func(self.writer.add_audio)


        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)
        self.current_note = None


        self.info = topleft_label()
        self.add_widget(self.info)


    def on_update(self):
        self.audio.on_update()


        self.info.text = (
            f"audio load: {self.audio.get_cpu_load():.2f}\n"
            f"mixer gain: {self.mixer.gain:.2f}\n"
            f"active generators: {self.mixer.get_num_generators()}\n"
            "keys: a/s/d/f/g/h/j=sine, q/w/e/r/t/y/u=square,\n"
            "1/2/3/4/5/6/7=triangle, v/b/n/m/,/./=sawtooth\n"
            "up/down=mixer gain, x=note_off latest, z=record toggle\n"
        )


    def on_key_down(self, keycode, modifiers):
        # Toggle recording
        if keycode[1] == "z":
            self.writer.toggle()
            return

        if keycode[1] == "up":
            self.mixer.set_gain(self.mixer.gain + 0.05)
            return

        if keycode[1] == "down":
            self.mixer.set_gain(self.mixer.gain - 0.05)
            return

        # Pick a sample rate if Audio exposes it; otherwise fall back to 44100
        sr = 44100
        if hasattr(self.audio, "get_sample_rate"):
            try:
                sr = self.audio.get_sample_rate()
            except Exception:
                sr = 44100

        pitches = [60, 62, 64, 65, 67, 69, 71]
        key_to_note = {
            "a": (pitches[0], 0.22, "sine"),
            "s": (pitches[1], 0.22, "sine"),
            "d": (pitches[2], 0.22, "sine"),
            "f": (pitches[3], 0.22, "sine"),
            "g": (pitches[4], 0.22, "sine"),
            "h": (pitches[5], 0.22, "sine"),
            "j": (pitches[6], 0.22, "sine"),
            "q": (pitches[0], 0.20, "square"),
            "w": (pitches[1], 0.20, "square"),
            "e": (pitches[2], 0.20, "square"),
            "r": (pitches[3], 0.20, "square"),
            "t": (pitches[4], 0.20, "square"),
            "y": (pitches[5], 0.20, "square"),
            "u": (pitches[6], 0.20, "square"),
            "1": (pitches[0], 0.24, "triangle"),
            "2": (pitches[1], 0.24, "triangle"),
            "3": (pitches[2], 0.24, "triangle"),
            "4": (pitches[3], 0.24, "triangle"),
            "5": (pitches[4], 0.24, "triangle"),
            "6": (pitches[5], 0.24, "triangle"),
            "7": (pitches[6], 0.24, "triangle"),
            "v": (pitches[0], 0.18, "sawtooth"),
            "b": (pitches[1], 0.18, "sawtooth"),
            "n": (pitches[2], 0.18, "sawtooth"),
            "m": (pitches[3], 0.18, "sawtooth"),
            ",": (pitches[4], 0.18, "sawtooth"),
            ".": (pitches[5], 0.18, "sawtooth"),
            "/": (pitches[6], 0.18, "sawtooth"),
        }

        # Start a note
        if keycode[1] in key_to_note:
            pitch, gain, wave = key_to_note[keycode[1]]
            note_gen = NoteGenerator(pitch=pitch, gain=gain, wave=wave, sample_rate=sr)
            env = Envelope(note_gen, attack_time=0.01, n1=1.0, decay_time=1.2, n2=1.0)
            self.mixer.add(env)
            self.current_note = note_gen
            return

        # Turn off the note
        if keycode[1] == "x":
            if self.current_note is not None:
                self.current_note.note_off()
            return




# part 5: Add code to this class for the creative part
class MainWidget2(BaseWidget):
    """
    Creative instrument: keyboard performance with timbre presets.

    Research idea:
    - Model woodwind-oriented families (oboe, clarinet, tenor sax, bassoon)
      plus a generic baseline using simple DSP parameters rather than exact
      physics.
    - Map each family to spectral and temporal descriptors:
      (1) harmonic balance, (2) envelope shape, (3) brightness filtering.

    Science / synthesis rationale:
    - Timbre perception depends strongly on overtone content and spectral slope.
      We approximate this with per-instrument harmonic envelopes calibrated from
      known woodwind tendencies (e.g., clarinet odd-harmonic dominance, flute
      fundamental-heavy spectrum, oboe brighter overtone distribution).
    - Transient shape also affects timbre identity. We set attack/decay and
      exponent parameters (n1, n2) to emulate articulation differences, e.g.
      short attacks for plucks and slower attacks for flute-like tones.
    - Many acoustic instruments behave like resonant low-pass systems where
      high frequencies decay faster. A one-pole filter
      y[n] = y[n-1] + alpha * (x[n] - y[n-1]) provides a lightweight proxy:
      lower alpha sounds warmer/darker, higher alpha sounds brighter.
    - Performance UX:
      Note keys stay fixed while meta keys switch timbre presets, so the same
      fingering yields different instrumental colors in real time.
    - Small "cheat" for recognizability:
      flute voices are transposed up 1 octave and bassoon voices are transposed
      down 1 octave so each preset sits closer to its typical playing register.
    """
    def __init__(self):
        super(MainWidget2, self).__init__()

        self.audio = Audio(1)
        self.writer = AudioWriter("data")
        self.audio.add_listen_func(self.writer.add_audio)
        self.mixer = Mixer()
        self.mixer.set_gain(0.18)
        self.audio.set_generator(self.mixer)
        self.current_note = None

        # C major across the home row for simple playability.
        self.note_keys = ["a", "s", "d", "f", "g", "h", "j", "k"]
        self.pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        self.key_to_pitch = dict(zip(self.note_keys, self.pitches))

        self.preset_order = ["generic", "oboe", "clarinet", "tenor_sax", "bassoon", "flute"]
        self.preset_idx = 0
        self.preset_name = self.preset_order[self.preset_idx]
        self.presets = {
            "generic": {
                # Baseline from parts 1-4 style: sine-like source + simple envelope.
                "harmonics": harmonic_amplitudes("sine", 12),
                "attack": 0.01,
                "decay": 1.2,
                "n1": 1.0,
                "n2": 1.0,
                "alpha": 1.0,  # alpha=1 behaves like passthrough (no LP darkening)
                "gain": 0.26,
                "pitch_offset": 0,
                "vibrato_rate_hz": 0.0,
                "vibrato_depth_semitones": 0.0,
            },
            "flute": {
                "harmonics": instrument_harmonics("flute", 12),
                "attack": 0.06,
                "decay": 1.9,
                "n1": 1.0,
                "n2": 1.1,
                "alpha": 0.11,
                "gain": 0.34,
                "pitch_offset": 12,
                "vibrato_rate_hz": 5.0,
                "vibrato_depth_semitones": 0.08,
            },
            "oboe": {
                "harmonics": instrument_harmonics("oboe", 12),
                "attack": 0.018,
                "decay": 1.45,
                "n1": 1.0,
                "n2": 0.88,
                "alpha": 0.46,  # brighter/nasal than sax
                "gain": 0.22,
                "pitch_offset": 5,  # slight upward register bias
                "vibrato_rate_hz": 5.8,
                "vibrato_depth_semitones": 0.05,
            },
            "clarinet": {
                "harmonics": instrument_harmonics("clarinet", 12),
                "attack": 0.02,
                "decay": 1.4,
                "n1": 1.0,
                "n2": 1.0,
                "alpha": 0.18,
                "gain": 0.24,
                "pitch_offset": 0,
                "vibrato_rate_hz": 0.0,
                "vibrato_depth_semitones": 0.0,
            },
            "tenor_sax": {
                "harmonics": instrument_harmonics("tenor_sax", 12),
                "attack": 0.042,
                "decay": 1.85,
                "n1": 1.15,
                "n2": 1.05,
                "alpha": 0.24,  # warmer than oboe
                "gain": 0.27,
                "pitch_offset": -2,  # lower register center
                "vibrato_rate_hz": 5.2,
                "vibrato_depth_semitones": 0.22,
            },
            "bassoon": {
                "harmonics": instrument_harmonics("bassoon", 12),
                "attack": 0.045,
                "decay": 1.8,
                "n1": 1.0,
                "n2": 1.2,
                "alpha": 0.14,
                "gain": 0.28,
                "pitch_offset": -12,
                "vibrato_rate_hz": 0.0,
                "vibrato_depth_semitones": 0.0,
            },
        }


        self.info = topleft_label()
        self.add_widget(self.info)

    def _cycle_preset(self, delta=1):
        self.preset_idx = (self.preset_idx + delta) % len(self.preset_order)
        self.preset_name = self.preset_order[self.preset_idx]

    def _set_preset(self, name):
        if name in self.presets:
            self.preset_name = name
            self.preset_idx = self.preset_order.index(name)

    def _make_voice(self, pitch, sample_rate):
        cfg = self.presets[self.preset_name]
        pitch = float(pitch) + float(cfg.get("pitch_offset", 0.0))
        note = MultiWaveNoteGenerator(
            pitch=pitch,
            gain=cfg["gain"],
            harmonic_profile=cfg["harmonics"],
            sample_rate=sample_rate,
            num_harmonics=12,
            vibrato_rate_hz=cfg.get("vibrato_rate_hz", 0.0),
            vibrato_depth_semitones=cfg.get("vibrato_depth_semitones", 0.0),
        )
        env = Envelope(
            note,
            attack_time=cfg["attack"],
            n1=cfg["n1"],
            decay_time=cfg["decay"],
            n2=cfg["n2"],
        )
        # Put the low-pass after the envelope for efficient timbre shaping.
        filt = OnePoleLowpass(env, alpha=cfg["alpha"])
        return note, filt

    def on_update(self):
        self.audio.on_update()
        cfg = self.presets[self.preset_name]
        self.info.text = (
            f"audio load: {self.audio.get_cpu_load():.2f}\n"
            f"preset: {self.preset_name} | alpha: {cfg['alpha']:.2f}\n"
            f"mix gain: {self.mixer.gain:.2f} | active voices: {self.mixer.get_num_generators()}\n"
            "note keys: a s d f g h j k\n"
            "switch timbre: shift(tenor sax), tab(flute), capslock(clarinet),\n"
            "               backspace(bassoon), enter(oboe), 0(generic)\n"
            "fallback switches: 0=generic, 1=flute, 2=oboe, 3=clarinet, 4=tenor sax, 5=bassoon, c=cycle\n"
            "controls: up/down mix gain, x note_off latest, z record toggle\n"
        )

    def on_key_down(self, keycode, modifiers):
        key = keycode[1]

        if key == "z":
            self.writer.toggle()
            return

        if key == "up":
            self.mixer.set_gain(self.mixer.gain + 0.03)
            return

        if key == "down":
            self.mixer.set_gain(self.mixer.gain - 0.03)
            return

        # Modifier-based timbre switches.
        if "shift" in modifiers:
            self._set_preset("tenor_sax")
        if key == "tab":
            self._set_preset("flute")
            return
        if key == "capslock":
            self._set_preset("clarinet")
            return
        if key == "backspace":
            self._set_preset("bassoon")
            return
        if key in ("enter", "numpadenter"):
            self._set_preset("oboe")
            return
        if key == "0":
            self._set_preset("generic")
            return

        # Reliable fallback switches (if meta keys vary by platform).
        if key == "1":
            self._set_preset("flute")
            return
        if key == "2":
            self._set_preset("oboe")
            return
        if key == "3":
            self._set_preset("clarinet")
            return
        if key == "4":
            self._set_preset("tenor_sax")
            return
        if key == "5":
            self._set_preset("bassoon")
            return
        if key == "c":
            self._cycle_preset(+1)
            return

        if key == "x":
            if self.current_note is not None:
                self.current_note.note_off()
            return

        if key in self.note_keys:
            pitch = self.key_to_pitch[key]
            sr = float(getattr(Audio, "sample_rate", 44100))
            note, voice = self._make_voice(pitch, sr)
            self.current_note = note
            self.mixer.add(voice)




# to run, on the command line, type: python pset1.py <num> to choose MainWidget 1 or 2
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"run {sys.argv[0]} <num> to choose MainWidget1 or MainWidget2")
    else:
        run(eval("MainWidget" + sys.argv[1])())
