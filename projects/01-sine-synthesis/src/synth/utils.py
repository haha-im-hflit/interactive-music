"""Part 1 — NoteGenerator (pitch, gain, note_off)
Goal: A generator that produces audio for a MIDI pitch (69 = A4 = 440 Hz)
at a given gain, and can stop when note_off() is called."""


A4_PITCH = 69
A4_FREQ = 440.0


def midi_to_hz(pitch: int) -> float:
    """
    Convert MIDI pitch number to frequency in Hz.
    MIDI 69 -> A4 -> 440 Hz.
    """
    return A4_FREQ * (2.0 ** ((pitch - A4_PITCH) / 12.0))
