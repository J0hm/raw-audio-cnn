import os
import csv
import pydub
from tqdm import tqdm
from math import ceil
import random
import numpy as np

set_splits = [0.6, 0.2, 0.2] # train test validate
block_len = 1
num_classes = 30
overlap_amt = 1/3
audio_dir = "rawaudio/"
dest_dir = "datasets/"
export_path = os.path.join(dest_dir, "len{}".format(block_len))
audio_path = os.path.join(export_path, "audio")
labels = sorted([subdir for subdir in os.listdir(audio_dir)])
print("Labels:", labels)

if not os.path.exists(audio_path):
   os.makedirs(audio_path)
   print("Directory", audio_path, "does not exist. Creating one now...")

# generates samples from the passed audio file
def sample_audio(audio, overlap, sample_length, sigma=0.1):
    duration = audio.duration_seconds 
    start = random.uniform(0, duration/10)
    start = 0
    samples = []
    # while we can still take a sample...
    while(start + sample_length <= duration):
        norm_overlap = max(0, min(np.random.normal(overlap, sigma), 1)) # clamp to [0,1] just in case...
        norm_overlap = overlap
        split = audio[start*1000:(start+sample_length)*1000]
        samples.append(split)
        start = start + sample_length*norm_overlap

    return samples

total_its = 0
hash_hits = 0

def optimize_bins_3(cap_a, cap_b, cap_c, weights, idx=0, seen=dict(), root=True, in_a=0, in_b=0, in_c=0, a=[], b=[], c=[]):
    global total_its, hash_hits 
    if(root):
        seen.clear()

    total_its += 1

    diffs = (cap_a-in_a, cap_b-in_b, cap_c-in_c)
    p = abs(diffs[0]) + abs(diffs[1]) + abs(diffs[2])

    if(len(a)+len(b)+len(c)==len(weights)): # base case
        return (p,a,b,c)
    else:
        opt = dict()
        e = weights[idx]
        ct = len(e)
        sizes = {'a': (in_a+ct, in_b, in_c),
                 'b': (in_a, in_b+ct, in_c),
                 'c': (in_a, in_b, in_c+ct)}
        lists = {'a': [a+[e], b, c],
                 'b': [a, b+[e], c],
                 'c': [a, b, c+[e]]}

        # here we will decide which branches to prioitize based on the one with the biggest diff (most empty space)
        diffdict = {key:diffs[idx] for idx, key in enumerate(('a', 'b', 'c'))}
        eval_order = [k for k, _ in sorted(diffdict.items(), key=lambda item: item[1], reverse=True)]

        for k in eval_order: 
            h = hash(sizes[k]) # hash inputs[k] to get the seen dict key
            if(h in seen):
                opt[k] = seen[h]
                hash_hits+=1
            else:
                opt[k] = optimize_bins_3(cap_a, cap_b, cap_c, weights, idx+1, seen, False, *sizes[k], *lists[k])
                seen[h] = opt[k]
            
            # short circuit hard 
            if(opt[k][0] < 3):                 
                return opt[k]

        return opt[min(opt, key=lambda key:opt[key][0])]

def flatten(lst):
    return [e for sl in lst for e in sl]

def export_samples(label_idx, label_name, samples, csvname, limit):
    with open(os.path.join(export_path, csvname), 'a+') as file:
        w = csv.writer(file)

        for idx, s in enumerate(samples):
            if(idx > limit):
                print("Stopping export: limit hit ({})".format(limit))
                break
            export_name = label_name+"_"+str(idx)+".wav" 
            w.writerow([export_name, label_idx])
            path = os.path.join(export_path, "audio", export_name)
            s.export(path, format="wav")

if __name__ == '__main__':
    sample_list = [[] for _ in range(num_classes)] # create sample store
    for label, subdir in enumerate(tqdm(labels)):
        target = labels.index(subdir)
        subdir_path = os.path.join(audio_dir, subdir)
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            audio = pydub.AudioSegment.from_wav(file_path)
            duration = audio.duration_seconds
            samples = sample_audio(audio, overlap_amt, block_len)
            if(len(samples) > 0):
                sample_list[label].append(sample_audio(audio, overlap_amt, block_len))
   
        data = sample_list[label]
        total = sum([len(sample) for sample in data])
        total_its, hash_hits, short_circuit_ct = 0,0,0
        print("Optimizing for list of {} audio files with {} total samples.".format(len(data), total))
        train_target, test_target, validate_target = [ceil(total*prop) for prop in set_splits]
        print("Targets:", train_target, test_target, validate_target)

        p, train, test, validate = optimize_bins_3(train_target, test_target, validate_target, data)
        print("Optimizer p:", p)
        print("File counts:", len(train), len(test), len(validate))
        print("Search: {} its, {} hash hits\n\n".format(total_its, hash_hits))
        assert len(train) + len(test) + len(validate) == len(data)
        assert sum(len(s) for s in train) + sum(len(s) for s in test) + sum(len(s) for s in validate) == total

        # limits are limits per label: we dont want one label with thousands
        export_samples(label, subdir, flatten(train), "train.csv", 300)
        export_samples(label, subdir, flatten(test), "test.csv", 100)
        export_samples(label, subdir, flatten(validate), "validate.csv", 100)




    
