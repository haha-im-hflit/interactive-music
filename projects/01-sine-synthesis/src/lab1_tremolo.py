import numpy as np


class Tremolo(object):
    """
    Tremolo effect generator.
    Takes an input generator and applies low-frequency amplitude modulation.
    """

    def __init__(self, input_gen, lfo_freq=5.0, depth=0.6, sample_rate=44100):
        super(Tremolo, self).__init__()
        self.input_gen = input_gen
        self.lfo_freq = float(lfo_freq)
        self.depth = float(np.clip(depth, 0.0, 1.0))
        self.sample_rate = int(sample_rate)
        self.frame = 0

    def set_input(self, input_gen):
        self.input_gen = input_gen

    def set_freq(self, lfo_freq):
        self.lfo_freq = float(lfo_freq)

    def set_depth(self, depth):
        self.depth = float(np.clip(depth, 0.0, 1.0))

    def generate(self, num_frames, num_channels):
        if self.input_gen is None:
            return np.zeros(num_frames, dtype=np.float32), True

        audio, continue_flag = self.input_gen.generate(num_frames, num_channels)
        audio = np.asarray(audio, dtype=np.float32)

        frames = np.arange(self.frame, self.frame + num_frames, dtype=np.float64)
        lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * self.lfo_freq * frames / self.sample_rate))

        # At depth=0, effect is bypassed (gain=1). At depth=1, full tremolo.
        mod = (1.0 - self.depth) + self.depth * lfo

        if num_channels > 1:
            mod = np.repeat(mod, num_channels)

        self.frame += num_frames
        return (audio * mod).astype(np.float32), continue_flag
