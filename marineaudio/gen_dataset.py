import os
import csv
from numpy.random import sample
import pydub
from torch import clamp
from tqdm import tqdm
from math import ceil, floor
import pandas as pd
import random
import numpy as np
import time
import sys

# idea for balancing dataset: make all possible segments and store them in their "bucket"
# then, select randomly from the buckets. remove selected item, continue until a bucket is empty

set_splits = [0.6, 0.2, 0.2] # train test validate
block_len = 1
num_classes = 30
overlap_amt = 1/3
audio_dir = "rawaudio/"
dest_dir = "datasets/"
#export_path = os.path.join(dest_dir, "len{}".format(block_len))
#audio_path = os.path.join(export_path, "audio")
#labels = sorted([subdir for subdir in os.listdir(audio_dir)])
#print("Labels:", labels)

#if not os.path.exists(audio_path):
#   os.makedirs(audio_path)
#   print("Directory", audio_path, "does not exist. Creating one now...")

# generates samples from the passed audio file
# overlap must be in [0, 1] and denotes the amount that two consecutive samples should overlap
# this is used to determine the next startpos
# pos_next = pos + (N(overlap, sigma)*sample_length)
def sample_audio(audio, overlap, sample_length, sigma=0.1):
    duration = audio.duration_seconds 
    #start = random.uniform(0, duration/10)
    start = 0
    samples = []
    # 0 .5 1 1.5 2 2.5 3
    # while we can still take a sample...
    while(start + sample_length <= duration):
        #norm_overlap = max(0, min(np.random.normal(overlap, sigma), 1)) # clamp to [0,1] just in case...
        norm_overlap = overlap
        split = audio[start*1000:(start+sample_length)*1000]
        samples.append(split)
        start = start + sample_length*norm_overlap

    return samples

# returns two lists of indexes, optimizing the total weight of list 1 to be ratio*total_weight and
# the total weight of list 2 to be (1-ratio)*total_weight


# DP time!
def optimize_bins(cap_a, cap_b, weights, in_a=0, in_b=0, idx=0, a=[], b=[], seen=dict(), root=True):
    if(root):
        seen.clear()
        in_a, in_b, idx = 0, 0, 0
        a, b = [], []

    p = abs(cap_a-in_a)+abs(cap_b-in_b)
    if(idx==len(weights)):
        return (p,a,b) 
    else:
        e = weights[idx]
        key_a = hash((in_a+e, in_b))
        key_b = hash((in_a, in_b+e))
        if(key_a in seen):
            opt_a = seen[key_a]
        else:
            opt_a = optimize_bins(cap_a, cap_b, weights, in_a+e, in_b, idx+1, a+[e], b, seen, root=False)
            seen[key_a] = opt_a
        if(key_b in seen):
            opt_b=seen[key_b]
        else:
            opt_b = optimize_bins(cap_a, cap_b, weights, in_a, in_b+e, idx+1, a, b+[e], seen, root=False)
            seen[key_b] = opt_b

        return opt_a if opt_a[0] < opt_b[0] else opt_b

def optimize_bins_3(cap_a, cap_b, cap_c, weights, in_a=0, in_b=0, in_c=0, idx=0, a=[], b=[], c=[], seen=dict(), root=True):
    if(root):
        seen.clear()

    p = abs(cap_a-in_a)+abs(cap_b-in_b)+abs(cap_c-in_c)

    if(idx==len(weights)):
        return (p,a,b,c) 
    else:
        opt = dict()
        e = weights[idx]
        key_a = hash((in_a+e, in_b, in_c))
        key_b = hash((in_a, in_b+e, in_c))
        key_c = hash((in_a, in_b, in_c+e))
        if(key_a in seen):
            opt['a'] = seen[key_a]
        else:
            opt['a'] = optimize_bins_3(cap_a, cap_b, cap_c, weights, in_a+e, in_b, in_c, idx+1, a+[e], b, c, seen, False)
            seen[key_a] = opt['a']
        if(key_b in seen):
            opt['b'] = seen[key_b]
        else:
            opt['b'] = optimize_bins_3(cap_a, cap_b, cap_c, weights, in_a, in_b+e, in_c, idx+1, a, b+[e], c, seen, False)
            seen[key_b] = opt['b']
        if(key_c in seen):
            opt['c'] = seen[key_c]
        else:
            opt['c'] = optimize_bins_3(cap_a, cap_b, cap_c, weights, in_a, in_b, in_c+e, idx+1, a, b, c+[e], seen, False)
            seen[key_c] = opt['c']


        return opt[min(opt, key=lambda key:opt[key][0])]




# ideas: cache already calculated P, indexed by (a, b)

# two options: add next elem to a or b
# index the DP table by the weight of a and b
# store the optimal weights so far

sys.setrecursionlimit(3200)
n_items = 90
data = [random.choice(range(10, 20)) for _ in range(n_items)]
cap = ceil(sum(data)/3)
start = time.time()
p, a, b, c = optimize_bins_3(cap, cap, cap, data)
end = time.time()
print("{}ms elapsed to do {} items".format((end-start)*1000, len(data)))
print(p, sum(a), sum(b), sum(c), len(a), len(b), len(c), a, b, c)

#sample_list = [[] for _ in range(num_classes)] # create sample store
#for label, subdir in enumerate(tqdm(labels)):
#    target = labels.index(subdir)
#    subdir_path = os.path.join(audio_dir, subdir)
#    for filename in os.listdir(subdir_path):
#        file_path = os.path.join(subdir_path, filename)
#        audio = pydub.AudioSegment.from_wav(file_path)
#        duration = audio.duration_seconds
#        samples = sample_audio(audio, overlap_amt, block_len)
#        if(len(samples) > 0):
#            sample_list[label].append(sample_audio(audio, overlap_amt, block_len))
    
#    total = sum([len(sample) for sample in sample_list[label]])
#    train_target, test_target, validate_target = [ceil(total*prop) for prop in set_splits]
#    p, train, other = optimize_bins(train_target, test_target+validate_target, sample_list[label])
    #total = sum([len(sample) for sample in other])
    #test_target, validate_target = ceil(total*set_splits[1]), ceil(total*set_splits[2])
    #print(total, other)
    #print([len(s) for s in other])
    #p2, test, validate = optimize_bins(test_target, validate_target, other)
#    print("Optimizer p1, p2:", p)
    #print("Lengths:", len(train), len(other), len(test), len(validate))
    
