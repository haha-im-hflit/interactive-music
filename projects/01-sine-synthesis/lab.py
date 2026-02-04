"""
6.101 Lab:
Audio Processing
"""

import os
import wave
import struct
# import typing  # optional import
# import pprint  # optional import

# No additional imports allowed!


def backwards(sound):
    """
    Returns a new sound containing the samples of the original in reverse
    order, without modifying the input sound.

    Args:
        sound: a dictionary representing the original mono sound

    Returns:
        A new mono sound dictionary with the samples in reversed order
    """
    # check using -k and -v that we just learned
    # -k lets u search certain words to run just those tests,
    # -v returns the names of tests more verbosely (v for verbose)

    return {
        "rate": sound["rate"],
        "samples": sound["samples"][::-1],
    }


def mix(sound1, sound2, p):
    """ mix 2 good sounds. p*sound1 + (1-p)*sound2.
    If rates differ or sound is malformed (as in samples not there), return none
    """
    if (
    "rate" not in sound1
    or "rate" not in sound2
    or "samples" not in sound1
    or "samples" not in sound2
    or sound1["rate"] != sound2["rate"]
    ):
        return None


    r = sound1["rate"]  # get rate
    sound1 = sound1["samples"]
    sound2 = sound2["samples"]

    if len(sound1) < len(sound2):
        length = len(sound1)
    elif len(sound2) < len(sound1):
        length = len(sound2)
    elif len(sound1) == len(sound2):
        length = len(sound1)
    else:
        print("whoops")
        return None

    mix_sound = []
    x = 0
    while x <= length:
        s2, s1 = p * sound1[x], sound2[x] * (1 - p)
        mix_sound.append(s1 + s2)  # add sounds
        x += 1
        if x == length:  # end
            break

    return {"rate": r, "samples": mix_sound}  # return new sound


def echo(sound, num_echoes, delay, scale):
    """
    Compute a new sound consisting of several scaled-down and delayed versions
    of the input sound. Does not modify input sound.

    Args:
        sound: a dictionary representing the original mono sound
        num_echoes: int, the number of additional copies of the sound to add
        delay: float, the amount of seconds each echo should be delayed
        scale: float, the amount by which each echo's samples should be scaled

    Returns:
        A new mono sound dictionary resulting from applying the echo effect.
    """
    rate = sound["rate"]
    original = sound["samples"]
    sample_delay = round(delay * sound["rate"])

    out_len = len(original) + num_echoes * sample_delay
    out_samples = [0.0] * out_len

    for i, v in enumerate(original):
        out_samples[i] += v #output contains og sound values/index

    for k in range(1, num_echoes + 1):
        add = scale ** k
        shift = k * sample_delay
        for i, v in enumerate(original):
            out_samples[i + shift] += add * v

    return {"rate": rate, "samples": out_samples}

def pan(sound):
    """
    Manipulate stereo sound by shifting volume in left and right speakers
    if we start with a stereo sound that is N samples:
    scale 1st sample in left channel by 1,
    2nd by 1-(1/N-1), 3rd by 1-(2/N-1), last by 0

    scale 1st sample in right channel by 0,
    2nd by(1/N-1), 3rd by (2/N-1), last by 1

    """
    rate = sound["rate"]
    left = sound["left"]
    right = sound["right"]
    N = len(left)

    if N == 0:
        return {"rate": rate, "left": [], "right": []}
    if N == 1:
        # First (and only) frame, left at full, right at zero
        return {"rate": rate, 'left': [left[0] * 1.0], "right": [right[0] * 0.0]}

    assert len(right) == N, "left/right channel lengths must match"
    denom = N - 1

    scale_left = [0.0] * N
    scale_right = [0.0] * N
    for i in range(N):
        t = i / denom  # 0 at start to 1.0 at end and reversed for other side
        scale_left[i] = left[i] * (1.0 - t)
        scale_right[i] = right[i] * t

    return {"rate": rate, "left": scale_left, "right": scale_right}

def remove_vocals(sound):
    """
    """
    if "left" not in sound or "right" not in sound:
        raise ValueError
    rate = sound["rate"]
    left = sound["left"]
    right = sound["right"]
    assert len(left) == len(right), "left/right channel lengths must match"

    samples = [left[i] - right[i] for i in range(len(left))]
    return {"rate": rate, "samples": samples}

# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds
#try zip function from removal


def load_wav(filename, stereo=False):
    """
    Given the filename of a WAV file, load the data from that file and return a
    Python dictionary representing that sound
    """
    file = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = file.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    if stereo:
        left = []
        right = []
        for _ in range(count):
            frame = file.readframes(1)
            if chan == 2:
                left.append(struct.unpack("<h", frame[:2])[0])
                right.append(struct.unpack("<h", frame[2:])[0])
            else:
                datum = struct.unpack("<h", frame)[0]
                left.append(datum)
                right.append(datum)

    if stereo:
        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = []
        for _ in range(count):
            frame = file.readframes(1)
            if chan == 2:
                left = struct.unpack("<h", frame[:2])[0]
                right = struct.unpack("<h", frame[2:])[0]
                samples.append((left + right) / 2)
            else:
                datum = struct.unpack("<h", frame)[0]
                samples.append(datum)

        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Given a dictionary representing a sound, and a filename, convert the given
    sound into WAV format and save it as a file with the given filename (which
    can then be opened by most audio players)
    """
    # make folders if they do not exist
    directory = os.path.realpath(os.path.dirname(filename))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for left, right in zip(sound["left"], sound["right"]):
            left = int(max(-1, min(1, left)) * (2**15 - 1))
            right = int(max(-1, min(1, right)) * (2**15 - 1))
            out.append(left)
            out.append(right)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/hello.wav, rather than just as hello.wav, to account for the
    # sound files being in a different directory than this file)
    hello = load_wav("sounds/hello.wav")

    write_wav(backwards(hello), "hello_reversed.wav")

    #5.1
    # Load original mystery sound finds from file
    mystery = load_wav("sounds/mystery.wav")

    # Reverse it using backwards function of mystery
    # (as explained in lab reverses pressure values to play sounds backwards)
    reversed_mystery = backwards(mystery)

    # Save reversed sound to  new file (test in terminal ls sounds)
    write_wav(reversed_mystery, "sounds/mystery_reversed.wav")


    #5.2
    synth = load_wav("sounds/synth.wav")
    water = load_wav("sounds/water.wav")

    p = 0.2
    mix_sound_sywa = mix(synth, water, p)
    if mix_sound_sywa is not None:
        write_wav(mix_sound_sywa, "sounds/mix_sound_sywa.wav")
        print("Saved/wrote sounds/mix_sound_sywa.wav")
    else:
        print("Mix failed (rate mismatch or malformed inputs).")

    #5.3
    chord = load_wav("sounds/chord.wav")
    chord_echo = echo(chord, num_echoes = 5, delay = 0.3, scale = 0.6)
    write_wav(chord_echo, "sounds/chord_echo.wav")
    print("Saved/wrote sounds/chord_echo.wav")

    #6.1
    car = load_wav("sounds/car.wav", stereo=True)
    car_pan = pan(car)
    write_wav(car_pan, "sounds/car_pan.wav")
    print("Saved/wrote sound/car_pan/wav")

    #6.2
    lookout_mountain = load_wav("sounds/lookout_mountain.wav", stereo=True)  # must load stereo
    lookout_mountain_no_voc = remove_vocals(lookout_mountain)
    write_wav(lookout_mountain_no_voc, "sounds/lookout_mountain_novocals.wav")
    print("Saved: sounds/lookout_mountain_novocals.wav")
