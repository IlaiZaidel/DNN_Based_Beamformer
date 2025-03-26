# create a csv to train the model

'''
speaker_1,az_1,elev_1,speaker_2,az_2,elev_2
'''

# az1 and az2 should be at least 30 degrees apart, elevs should be in range [-30,30]
import numpy as np
from pathlib import Path
import random
from glob import glob
from tqdm import tqdm
import random
from random import shuffle
import pandas as pd
root_path = Path('/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/')
output_csv = Path("/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/output_paths_HRTF_extraction_librispeech_en_3k_test_sir_U_0_5.csv")

columns = [
    "beta", "room_x", "room_y", "room_z",
    "mic_x", "mic_y", "mic_z",
    "speaker_x", "speaker_y", "speaker_z",
    "noise1_x", "noise1_y", "noise1_z",
    "noise2_x", "noise2_y", "noise2_z",
    "angleOrientation", "angle_x", "angle_n1", "angle_n2", "radius"
]

df = pd.DataFrame(columns=columns)
speakers ={}
speakers_list = glob(str(root_path/'**'))
j=0
max_samples = 3000
for i in tqdm(range(4)):
    for speaker in speakers_list:
        name = speaker.split('/')[-1]
        l = glob(speaker+'/**/*.wav')
        if len(l)>1:
            shuffle(l)
            speakers[name] = l

    while len(speakers.keys()) >=2 and j<max_samples:
        speaker = random.sample(list(speakers.keys()),1)[0]
        s1 = speakers[speaker].pop()
        if len(speakers[speaker])<1:
            speakers.pop(speaker)
        rnd_speaker = random.sample(list(speakers.keys()),1)[0]
        while rnd_speaker == speaker:
            rnd_speaker = random.sample(list(speakers.keys()),1)[0]
        s2 = speakers[rnd_speaker].pop()
        if len(speakers[rnd_speaker])<1:
            speakers.pop(rnd_speaker)

        # 3 speakers    
        # rnd_speaker2 = random.sample(list(speakers.keys()),1)[0]
        # while rnd_speaker2 == speaker or rnd_speaker2 ==rnd_speaker :
        #     rnd_speaker2 = random.sample(list(speakers.keys()),1)[0]
        # s3 = speakers[rnd_speaker2].pop()
        # if len(speakers[rnd_speaker2])<1:
        #     speakers.pop(rnd_speaker2)

        #az and elev:

        az1 = random.choice(list(range(0, 91)) + list(range(270, 361)))

        # Generate the second integer such that it is at least 30 apart from the first
        if 0 <= az1 <= 90:
            valid_ranges = list(range(0, max(1, az1 - 50))) + list(range(min(90, az1 + 50) + 1, 91))
        else:  # 270 <= az1 <= 360
            valid_ranges = list(range(270, max(271, az1 - 50))) + list(range(min(360, az1 + 50) + 1, 361))

        az2 = random.choice(valid_ranges)
        elev1,elev2 = [random.randint(-10, 10) for _ in range(2)]
        sir =  np.random.uniform(0,5,1)[0]
        df.loc[len(df)] = [s1,az1,elev1,s2,az2,elev2,sir]
        j+=1
df.to_csv(output_csv)
