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
export_path = os.path.join(dest_dir, "len{}".format(block_len))
audio_path = os.path.join(export_path, "audio")
labels = sorted([subdir for subdir in os.listdir(audio_dir)])
print("Labels:", labels)

if not os.path.exists(audio_path):
   os.makedirs(audio_path)
   print("Directory", audio_path, "does not exist. Creating one now...")

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
total_its = 0
hash_hits = 0
short_circuit_ct = 0

# TODO implement short circuting when in goes over cap.
# ISSUE: must deal with the case where the optimal is subbranch of one of the short-circuited ones
# this causes problems since we are populating our seen hashtable with incomplete sets,
# and we sometimes get incomplete results.
# maybe instead of short circuiting, set calculate and set P to be very large when it is evaluated?
# i.e. at the opt[k] = optimize_bins_3(...) on line 100
# this would require calculating P again, but that should be pretty quick
def optimize_bins_3(cap_a, cap_b, cap_c, weights, idx=0, seen=dict(), root=True, in_a=0, in_b=0, in_c=0, a=[], b=[], c=[]):
    global total_its, hash_hits, short_circuit_ct
    if(root):
        seen.clear()

    total_its += 1

    diffs = (cap_a-in_a, cap_b-in_b, cap_c-in_c)
    p = abs(diffs[0]) + abs(diffs[1]) + abs(diffs[2])

    if(len(a)+len(b)+len(c)==len(weights)):
        return (p,a,b,c)
    else:
        opt = dict()
        e = weights[idx]
        sizes = {'a': (in_a+e, in_b, in_c),
                 'b': (in_a, in_b+e, in_c),
                 'c': (in_a, in_b, in_c+e)}
        lists = {'a': [a+[e], b, c],
                 'b': [a, b+[e], c],
                 'c': [a, b, c+[e]]}

        # here we will decide which branches to prioitize based on the one with the biggest diff (most empty space)
        diffdict = {key:diffs[idx] for idx, key in enumerate(('a', 'b', 'c'))}
        eval_order = [k for k, _ in sorted(diffdict.items(), key=lambda item: item[1], reverse=True)]

        for k in eval_order: # 'a', 'b', 'c'
            h = hash(sizes[k]) # hash inputs[k] to get the seen dict key
            if(h in seen):
                opt[k] = seen[h]
                hash_hits+=1
            else:
                opt[k] = optimize_bins_3(cap_a, cap_b, cap_c, weights, idx+1, seen, False, *sizes[k], *lists[k])
                seen[h] = opt[k]
            
            # short circuit hard if opt = 0, 1, 2
            # < num bins is the actual heuristic: with large enough datasets the optimal P will be mod number of bins, so we take any solution that could be the optimal as optimal
            # this prevents us from searching the entire tree for essentially no gain
            if(opt[k][0] < 3):                 
                return opt[k]

        return opt[min(opt, key=lambda key:opt[key][0])]



def test_func():
    n_items = 10000
    sys.setrecursionlimit(20000)
    its_sum = 0
    testct = 50
    for _ in range(testct):
        total_its, hash_hits, short_circuit_ct = 0,0,0
        data = [random.choice(range(10, 100)) for _ in range(n_items)]
        cap = ceil(sum(data)/3)
        start = time.time()
        p, a, b, c = optimize_bins_3(cap, cap, cap, data)
        end = time.time()
        print("{}ms elapsed to do {} items".format((end-start)*1000, len(data)))
        print(p, sum(a), sum(b), sum(c), len(a), len(b), len(c))
        print((sum(a)+sum(b)+sum(c)==sum(data)), len(a)+len(b)+len(c)==len(data))
        print(total_its, hash_hits, short_circuit_ct)
        its_sum += total_its
    print("Average iteratiosn: {}".format(its_sum/testct))




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
   
    data = [len(sample) for sample in sample_list[label]]
    total = sum(data)
    total_its, hash_hits, short_circuit_ct = 0,0,0
    print("Optimizing for list of {} audio files with {} total samples.".format(len(data), total))
    train_target, test_target, validate_target = [ceil(total*prop) for prop in set_splits]
    print("Targets:", train_target, test_target, validate_target)
    p, train, test, validate = optimize_bins_3(train_target, test_target, validate_target, data)
    print("Optimizer p:", p)
    print("Sums:", sum(train), sum(test), sum(validate))
    print("{} its, {} hits, {} short circuit".format(total_its, hash_hits, short_circuit_ct))
    assert len(train) + len(test) + len(validate) == len(data)
    
