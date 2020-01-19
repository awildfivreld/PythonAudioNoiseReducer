import sys
import os
import noisereduce as nr
import subprocess
from scipy.io import wavfile
import matplotlib
import numpy as np
import tempfile

max_ram_usage = 8
max_part_len = int(1000*(max_ram_usage*1250)) # Uses about 4-5 gigs of memory 
SYSTEM_SEPARATOR = "\\"

if ("--help" in sys.argv or "-h" in sys.argv):
    print("""
        Reduce noise from a video clip.
        
        Arguments cannot be combined in one (ie. -iov)
        -i --input: Specify input video to be reduced.
        -o --output: Specify output video file. Default: input video + "_reduced"
        -v --verbose: Output program steps
        --memory_limit: Specify about how much memory the program is allowed to use. Default ~5
        --ffmpeg_path: If ffmpeg is not on PATH, specify location. 
    """)

INPUT_FILE = ""
if ("-i" in sys.argv or "--input" in sys.argv):
    try:
        INPUT_FILE = sys.argv[sys.argv.index("--input")+1]
    except ValueError:
        pass
    try:
        INPUT_FILE = sys.argv[sys.argv.index("-i")+1]
    except ValueError:
        pass
else:
    print("Input file must be specified with -i or --input")

OUTPUT_FILE = ""
if ("-o" in sys.argv or "--output" in sys.argv):
    try:
        OUTPUT_FILE = sys.argv[sys.argv.index("--output")+1]
    except ValueError:
        pass
    try:
        OUTPUT_FILE = sys.argv[sys.argv.index("-o")+1]
    except ValueError:
        pass
else:
    OUTPUT_FILE = INPUT_FILE[:INPUT_FILE.find(".")] + "_reduced" + INPUT_FILE[INPUT_FILE.find("."):]

if ("-v" in sys.argv or "--verbose" in sys.argv):
    pass
if ("--low_memmory" in sys.argv):
    pass
if ("--ffmpeg_path" in sys.argv):
    pass

class NoiseReduceVideo:
    def __init__(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = tmpdir
            self.load_audio()
            self.denoise_audio()

    def load_audio(self):
        subprocess.check_output(["bin{}ffmpeg.exe".format(SYSTEM_SEPARATOR), "-i", INPUT_FILE, "-vn", "{}{}vidaud.wav".format(self.tmpdir, SYSTEM_SEPARATOR), "-y"])
        subprocess.check_output(["bin{}ffmpeg.exe".format(SYSTEM_SEPARATOR), "-sseof", "-64", "-i", INPUT_FILE , "{}{}noise.wav".format(self.tmpdir, SYSTEM_SEPARATOR), "-y"])
        self.rate1, self.data1 = wavfile.read("{}{}vidaud.wav".format(self.tmpdir, SYSTEM_SEPARATOR))
        self.rate2, self.data2 = wavfile.read("{}{}noise.wav".format(self.tmpdir, SYSTEM_SEPARATOR))

    def denoise_audio(self):
        data1 = self.data1 / 32768
        data2 = self.data2 / 32768
        concatlist = []

        for i, part_len in enumerate(range(0, len(data1), max_part_len)):
            print("Proccesing part {}, {} -> {}".format(i, part_len/self.rate1/60, min(part_len + max_part_len, len(data1))/self.rate1/60))
            data = data1[part_len:part_len + max_part_len]
            old_shape = data.shape
            processed_data = nr.reduce_noise(audio_clip=data.flatten(), noise_clip=data2.flatten(), use_tensorflow=False)
            wavfile.write("{}{}part{}_red.wav".format(self.tmpdir, SYSTEM_SEPARATOR, i), self.rate1, processed_data.reshape(old_shape))
            concatlist.append("file '{}{}part{}_red.wav'".format(self.tmpdir, SYSTEM_SEPARATOR, i))

        with open("{}{}concatlist.txt".format(self.tmpdir, SYSTEM_SEPARATOR), "w+") as clist:
            clist.write("\n".join(concatlist))


        subprocess.check_output(["bin{}ffmpeg".format(SYSTEM_SEPARATOR), "-f", "concat", "-safe", "0", "-i", "{}{}concatlist.txt".format(self.tmpdir, SYSTEM_SEPARATOR), "-c", "copy", "{}{}data1_red.wav".format(self.tmpdir, SYSTEM_SEPARATOR), "-y"])
        subprocess.check_output(["bin{}ffmpeg".format(SYSTEM_SEPARATOR), "-i", INPUT_FILE, "-i", "{}{}data1_red.wav".format(self.tmpdir, SYSTEM_SEPARATOR),"-map", "0:v?", "-vcodec", "copy", "-map", "1:a", OUTPUT_FILE, "-y"])


test = NoiseReduceVideo()
