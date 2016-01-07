import os
import sys
import tables
import random
import fnmatch
import numpy as np
from scipy.io import wavfile
from features import mfcc

h5_file_path = "/ha/work/people/klejch/saved_data.h5"
data_path = "/a/merkur2/vystadial/speech-corpora/en/TIMIT_Acoustic-Phonetic_Cont.Speech_Corpus_LDC93S1/timit/TIMIT/TRAIN/"

phones = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',
    'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',
    'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
    'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',
    'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
    'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
    'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
    'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
    'v', 'w', 'y', 'z', 'zh'
]

phonesToInt = dict(zip(phones, range(len(phones))))

def read_phones_transcript(wav_path):
    transcript = wav_path[:-4] + ".PHN"

    phones = []
    with open(transcript, 'r') as f:
        for line in f:
            phones.append(phonesToInt[line.strip().split(' ')[-1]])

    return phones

def read_timit_wav(wav_path):
    tmp_path = "/ha/work/people/klejch/test.wav"
    os.system("sox %s -V3 -t sndfile %s 2> /dev/null" % (wav_path, tmp_path))
    _, d = wavfile.read(tmp_path)
    os.remove(tmp_path)
    return d

def wav_paths_from_scp(scp_path):
    with open(scp_path) as f:
        return [line.strip().split(" ")[4] for line in f.readlines()]

# http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
h5_file = tables.openFile(h5_file_path, mode='w')

data = [
    ('train', '/ha/work/people/klejch/kams/kaldi/egs/timit/s5/data/train/wav.scp'),
    ('validate', '/ha/work/people/klejch/kams/kaldi/egs/timit/s5/data/test/wav.scp'),
    ('test', '/ha/work/people/klejch/kams/kaldi/egs/timit/s5/data/dev/wav.scp')
]

for (name, scp_path) in data:
    data_x = h5_file.createVLArray(h5_file.root, '%s_data_x' % name, tables.Float32Atom(shape=()), filters=tables.Filters(1))
    data_x_shapes = h5_file.createVLArray(h5_file.root, '%s_data_x_shapes' % name, tables.Int32Atom(shape=()), filters=tables.Filters(1))
    data_y = h5_file.createVLArray(h5_file.root, '%s_data_y' % name, tables.Int32Atom(shape=()), filters=tables.Filters(1))

    for wav_path in wav_paths_from_scp(scp_path):
        phones = read_phones_transcript(wav_path)
        data_y.append(np.array(phones, dtype='int32'))

        audio_data = mfcc(read_timit_wav(wav_path), 16000)
        data_x_shapes.append(np.array(audio_data.shape, dtype='int32'))
        data_x.append(audio_data.astype('float32').flatten())
        print "%s file processed" % wav_path

h5_file.close()
