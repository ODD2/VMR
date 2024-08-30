

import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
from mv_dts import MusicVideoDataset


# vids = glob("/scratch4/users/od/YTCharts/videos/*.mp4")
# print(vids)


def runner(data):

    try:
        label = data["labels"].item()
        pred = -1
        start_time = data["timestamps"][0].item()
        end_time = data["timestamps"][-1].item()
        scene_list = detect(
            data["video_path"],
            AdaptiveDetector(),
            start_time=start_time,
            end_time=end_time
        )
        if (len(scene_list) == 0):
            pred = 2
        elif float(scene_list[0][1]) - start_time < 5:
            pred = 0
        else:
            pred = 1
        return pred, label
    except Exception as e:
        print(f"failed to process {data['video_path']} with {e}")


if __name__ == '__main__':
    dataset = MusicVideoDataset(
        "test",
        frames=10,
        t_res=3,
        duration=10,
        portion=1.0,
        offset_ratio=0.2,
        meta_only=True,
        dts_ver="v1"
    )

    samples = len(dataset)
    preds = []
    labels = []
    with Pool(16) as p:
        for result in tqdm(p.imap_unordered(runner, [dataset[i] for i in range(samples)]), total=samples):
            preds.append(result[0])
            labels.append(result[1])

    preds = np.array(preds)
    labels = np.array(labels)
    print(f"Accuracy: {np.sum(preds == labels) / len(labels)}")
