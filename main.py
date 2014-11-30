#coding: utf-8

import os
import sys
import itertools
import numpy as np
import bisect
from yaafelib import Engine, AudioFileProcessor, FeaturePlan


RES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'resources'
)
WAV_PATH = os.path.join(RES_DIR, 'DTMF_dialing.wav')
ANS_PATH = os.path.join(RES_DIR, 'answer.txt')
SILENT_MAG_THRESHOLD = 10
KEYS = [
    '1', '2', '3', 'A',
    '4', '5', '6', 'B',
    '7', '8', '9', 'C',
    '*', '0', '#', 'D'
]
SILENT_KEY = 'S'
UNKNOWN_KEY = 'U'
LOWER_FREQS = [697, 770, 852, 941]
UPPER_FREQS = [1209, 1336, 1477, 1633]


def purge_seq(seq):
    rst = []
    for k, vs in itertools.groupby(seq):
        if k == SILENT_KEY or k == UNKNOWN_KEY:
            if len(rst) == 0:
                continue
            else:
                rst.append("")
        else:
            if len(rst) == 0:
                rst.append("")
            rst[-1] += k * ((len(list(vs)) + 1) / 2)  # XXX
    if rst[-1] == "":
        rst = rst[:-1]
    return rst


def detect(wav_path, ans_path=None):
    sample_rate = 8000
    block_size = 1024
    step_size = block_size / 2
    n_band = block_size / 2
    freq_bound = [i * sample_rate / 2. / n_band for i in range(n_band + 1)]

    plan = FeaturePlan(sample_rate=sample_rate, resample=True)
    plan.addFeature(
        'power_spectrum: PowerSpectrum blockSize=%d stepSize=%d' % (
            block_size, step_size
        )
    )
    dataflow = plan.getDataFlow()
    afp = AudioFileProcessor()
    engine = Engine()
    engine.load(dataflow)
    afp.processFile(engine, wav_path)
    spectrogram = engine.readOutput('power_spectrum')
    seq = []
    for spectrum in spectrogram:
        mean_mag = np.mean(spectrum)
        if mean_mag <= SILENT_MAG_THRESHOLD:
            seq.append(SILENT_KEY)
            continue
        lower_data = (-1, -1)
        upper_data = (-1, -1)

        for target_idx, target_freq in itertools.chain(
            enumerate(LOWER_FREQS), enumerate(UPPER_FREQS)
        ):
            idx = bisect.bisect(freq_bound, target_freq)
            assert idx > 0
            freq1 = freq_bound[idx - 1]
            mag1 = spectrum[idx - 1]
            freq2 = freq_bound[idx]
            mag2 = spectrum[idx]
            w1 = 1. * (freq2 - target_freq) / (freq2 - freq1)
            w2 = 1. - w1
            target_mag = (w1 * mag1 + w2 * mag2)

            if target_mag > mean_mag * 2:
                if target_freq < 1000:
                    if target_mag > lower_data[1]:
                        lower_data = (target_idx, target_mag)
                else:
                    if target_mag > upper_data[1]:
                        upper_data = (target_idx, target_mag)

        lower_idx = lower_data[0]
        upper_idx = upper_data[0]
        if lower_idx == -1 or upper_idx == -1:
            seq.append(UNKNOWN_KEY)
        else:
            seq.append(KEYS[lower_idx * len(LOWER_FREQS) + upper_idx])

    ans = purge_seq(seq)
    if ans_path is not None:
        with open(ans_path) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                assert line == ans[i], "%s != %s" % (line, ans[i])
                print '[%d] %s' % (i, line)
    else:
        for i, line in enumerate(ans):
            print '[%d] %s' % (i, line)


def main(argv):
    if len(argv) == 1:
        detect(WAV_PATH, ANS_PATH)
    else:
        detect(*argv[1:])


if __name__ == '__main__':
    main(sys.argv)
