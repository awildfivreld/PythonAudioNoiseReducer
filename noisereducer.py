import sys
import os
import noisereduce as nr
import subprocess
from scipy.io import wavfile
import matplotlib
import numpy as np
import tempfile
import shutil

max_ram_usage = 5.5
max_part_len = int(1818181*max_ram_usage) # Uses about 4-5 gigs of memory 

class NoiseReduceVideo:
    def __init__(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = tmpdir
            self.load_audio()
            self.denoise_audio()

    def load_audio(self):
        subprocess.check_output(["bin\\ffmpeg.exe", "-i", "combined.mp4", "-vn", self.tmpdir+"\\vidaud.wav", "-y"])
        subprocess.check_output(["bin\\ffmpeg.exe", "-sseof", "-64", "-i", "combined.mp4", self.tmpdir+"\\noise.wav", "-y"])
        self.rate1, self.data1 = wavfile.read(self.tmpdir+"\\vidaud.wav")
        self.rate2, self.data2 = wavfile.read(self.tmpdir+"\\noise.wav")
    
    def denoise_audio(self):
        data1 = self.data1 / 32768
        data2 = self.data2 / 32768
        old_shape = data1.shape
        concatlist = []

        for i, part_len in enumerate(range(0, len(data1), max_part_len)):
            print("Proccesing part {}, {} -> {}".format(i, part_len, part_len + max_part_len))
            data = data1[part_len:part_len + max_part_len]
            processed_data = nr.reduce_noise(audio_clip=data.flatten(), noise_clip=data2.flatten())
            wavfile.write(self.tmpdir+"\\part{}_red.wav".format(i), self.rate1, processed_data.reshape(len(data), 2))
            concatlist.append("file '{}\\part{}_red.wav'".format(self.tmpdir, i))

            if (i == 1):
                break

        with open(self.tmpdir+"\\concatlist.txt", "w+") as clist:
            clist.write("\n".join(concatlist))


        subprocess.check_output(["bin\\ffmpeg", "-f", "concat", "-safe", "0", "-i", self.tmpdir+"\\concatlist.txt", "-c", "copy", self.tmpdir+"\\data1_red.wav", "-y"])
        subprocess.check_output(["bin\\ffmpeg", "-i", "combined.mp4", "-i", self.tmpdir+"\\data1_red.wav", "-map", "0:v", "-vcodec", "copy", "-map", "1:a", "output.mp4", "-y"])


test = NoiseReduceVideo()
