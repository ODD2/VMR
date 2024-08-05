import os
import json
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg


vids = glob("/scratch4/users/od/YTCharts/v3/*/*.mp4")
print(vids)


def runner(video_file):

    try:
        json_file = video_file.replace(".mp4", ".json")
        scene_list = detect(video_file, AdaptiveDetector())
        scenes = len(scene_list)
        with open(json_file, "r") as f:
            meta = json.load(f)
        meta["scenes"] = scenes
        with open(json_file, "w") as f:
            json.dump(meta, f)
    except Exception as e:
        print(f"failed to process {video_file} with {e}")


if __name__ == '__main__':

    with Pool(16) as p:
        for i in tqdm(
                p.imap_unordered(
                    runner,
                    [vid for vid in vids]
                ),
                total=len(vids)
        ):
            continue
