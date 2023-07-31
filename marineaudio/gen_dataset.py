import os
import csv
import pydub
from tqdm import tqdm
from math import ceil, floor
import pandas as pd

set_splits = [0.7, 0.15, 0.15] # train test validate
block_len = 2
max_samples = 250
audio_dir = "rawaudio/"
dest_dir = "datasets/"
export_path = os.path.join(dest_dir, "len{}".format(block_len))
audio_path = os.path.join(export_path, "audio")
labels = sorted([subdir for subdir in os.listdir(audio_dir)])
print("Labels:", labels)

if not os.path.exists(audio_path):
   os.makedirs(audio_path)
   print("Directory", audio_path, "does not exist. Creating one now...")

with open(os.path.join(export_path, "annotations.csv"), 'w') as csvfile:
    writer = csv.writer(csvfile)
    for subdir in tqdm(labels):
        target = labels.index(subdir)
        subdir_path = os.path.join(audio_dir, subdir)
        samples=0
        for filename in os.listdir(subdir_path):
            if samples > max_samples: break
            file_path = os.path.join(subdir_path, filename)
            audio = pydub.AudioSegment.from_wav(file_path)
            duration = audio.duration_seconds
            for start in range(0, ceil(duration), block_len):
                split = audio[start*1000:(start+block_len)*1000]
                export_name = subdir+filename.split(".")[0]+"_"+str(start)+".wav" 
                path = os.path.join(export_path, "audio", export_name)
                split.export(path, format="wav")
                writer.writerow([export_name, target])
                samples += 1
        

annotations = pd.read_csv(os.path.join(export_path, "annotations.csv"), header=None, index_col=0)
print("Generating train and test sets...")
count = annotations.size
train = annotations.sample(n=floor(count*set_splits[0]))
test = annotations.sample(n=floor(count*set_splits[1]))
validate = annotations.sample(n=floor(count*set_splits[2]))
train.to_csv(os.path.join(export_path, "train.csv"), header=False)
test.to_csv(os.path.join(export_path, "test.csv"), header=False)
validate.to_csv(os.path.join(export_path, "validate.csv"), header=False)

print("Train: ", train.size, "samples.")
print("Test: ", test.size, "samples.")
print("Validate: ", validate.size, "samples.")



