import os
import json


SPLIT = "test"
VER_NAME = os.path.join("v1", SPLIT)
BASE_FOLDER = "/scratch4/users/od/YTCharts"
SPLIT_FILE = f"/scratch4/users/od/YTCharts/splits/{SPLIT}.txt"
VER_FOLDER = os.path.join(BASE_FOLDER, VER_NAME)


def get_video_path(id):
    video_with_ext = id + ".mp4"
    return video_with_ext, os.path.join(BASE_FOLDER, "videos", video_with_ext)


def get_audio_path(id):
    audio_with_ext = id + ".mp3"
    return audio_with_ext, os.path.join(BASE_FOLDER, "separated/htdemucs", id, "no_vocals.mp3")


def get_meta_path(id):
    meta_with_ext = id + ".json"
    return meta_with_ext, os.path.join(BASE_FOLDER, "metas", meta_with_ext)


def main():
    # GET IDS
    ids = []
    with open(SPLIT_FILE, "r") as f:
        for line in f:
            ids.append(line.strip())

    # CONDITIONING
    _ids = []
    for id in ids:
        with open(get_meta_path(id)[1], "r") as f:
            try:
                meta = json.load(f)
                if meta["scenes"] >= 20:
                    _ids.append(id)
            except Exception as e:
                print(id, e)
    ids = _ids

    # CREATE LINKS
    os.makedirs(VER_FOLDER, exist_ok=True)

    os.chdir(VER_FOLDER)

    for id in ids:
        video = get_video_path(id)
        audio = get_audio_path(id)
        meta = get_meta_path(id)
        os.system("ln -s {} ./{}".format(os.path.relpath(video[1], VER_FOLDER), video[0]))
        os.system("ln -s {} ./{}".format(os.path.relpath(audio[1], VER_FOLDER), audio[0]))
        os.system("ln -s {} ./{}".format(os.path.relpath(meta[1], VER_FOLDER), meta[0]))


if __name__ == "__main__":
    main()
