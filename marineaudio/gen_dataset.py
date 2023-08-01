import os
import csv
import pydub
from tqdm import tqdm
from math import ceil, floor
import pandas as pd
import random
import numpy as np

# idea for balancing dataset: make all possible segments and store them in their "bucket"
# then, select randomly from the buckets. remove selected item, continue until a bucket is empty

set_splits = [0.6, 0.2, 0.2] # train test validate
block_len = 1
max_samples = 250
num_classes = 30
audio_dir = "rawaudio/"
dest_dir = "datasets/"
export_path = os.path.join(dest_dir, "len{}".format(block_len))
audio_path = os.path.join(export_path, "audio")
labels = sorted([subdir for subdir in os.listdir(audio_dir)])
print("Labels:", labels)

if not os.path.exists(audio_path):
   os.makedirs(audio_path)
   print("Directory", audio_path, "does not exist. Creating one now...")

sample_list = [[] for _ in range(num_classes)] # create sample store
for label, subdir in enumerate(tqdm(labels)):
    target = labels.index(subdir)
    subdir_path = os.path.join(audio_dir, subdir)
    for filename in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, filename)
        audio = pydub.AudioSegment.from_wav(file_path)
        duration = audio.duration_seconds
        for start in range(0, ceil(duration), block_len):
            split = audio[start*1000:(start+block_len)*1000]
            if(len(sample_list[label]) > max_samples): break
            if(split.duration_seconds >= block_len/2):
                export_name = subdir+filename.split(".")[0]+"_"+str(start)+".wav" 
                path = os.path.join(export_path, "audio", export_name)
                split.export(path, format="wav")
                sample_list[label].append(export_name)
    print("Label {} ({}) has {} samples.".format(subdir, label, len(sample_list[label])))

# splits the list of samples into train/test/validate 
with open(os.path.join(export_path, "train.csv"), 'w') as trainfile, open(os.path.join(export_path, "test.csv"), 'w') as testfile, open(os.path.join(export_path, "validate.csv"), 'w') as validatefile: 
    train_writer = csv.writer(trainfile)
    test_writer = csv.writer(testfile)
    validate_writer = csv.writer(validatefile)

    for label, _ in enumerate(tqdm(labels)):
        train_samples = random.sample(sample_list[label], floor(len(sample_list[label])*set_splits[0]))
        test_samples = random.sample(sample_list[label], floor(len(sample_list[label])*set_splits[1]))
        validate_samples = random.sample(sample_list[label], floor(len(sample_list[label])*set_splits[2]))
        
        train_writer.writerows(zip(train_samples, [label]*len(train_samples)))
        test_writer.writerows(zip(test_samples, [label]*len(test_samples)))
        validate_writer.writerows(zip(validate_samples, [label]*len(validate_samples)))
