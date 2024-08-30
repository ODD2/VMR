import os
import json
import torch
import numpy as np
import torchvision
import librosa
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader


def concat_segments(segments):
    _segments = [dict(label="__holder__", start=0, beg=0)]
    for i, seg in enumerate(segments):
        if not (_segments[-1]["label"] == seg["label"]):
            _segments[-1]["end"] = seg["start"]
            _segments.append(seg)
        elif i == len(segments) - 1:
            _segments[-1]["end"] = seg["end"]
        else:
            continue
    _segments.pop(0)
    return _segments


def fetch_frames(video_path, duration, offset=0, num_frames=10, transform=(lambda x: x)):
    frames = torchvision.io.read_video(video_path, offset, offset + duration, 'sec', 'TCHW')[0]
    if (num_frames > 1):
        stride = (frames.shape[0] - 1) / (num_frames - 1)  # number of video frames
    else:
        stride = 0
    frames = frames[[int(i * stride) for i in range(num_frames)]]
    assert frames.shape[0] == num_frames, "number of frames is not correct"
    frames = transform(frames)
    return frames


class MusicVideoDataset(Dataset):

    def __init__(
        self,
        split,
        frames=16,
        duration=30,
        t_res=5,
        px=336,
        portion=1.0,
        offset_ratio=0.2,
        reverse=False,
        with_audio=False,
        debug=False,
        meta_only=False,
        dts_ver="v1"
    ):
        super().__init__()
        self.split = split
        self.frames = frames
        self.t_res = t_res
        self.major_path = f"/scratch4/users/od/YTCharts/{dts_ver}/{split}/"
        self.analy_path = f"/scratch4/users/od/YTCharts/analysis/"
        self.duration = duration
        self.with_audio = with_audio
        self.debug = debug
        self.meta_only = meta_only
        self.offset_ratio = offset_ratio
        self.entity_list = sorted(
            [
                os.path.basename(file)[:-4]
                for file in glob(f"{self.major_path}/*.mp3")
            ],
            reverse=reverse
        )
        self.entity_list = self.entity_list[:int(len(self.entity_list) * portion)]
        self.img_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (px, px),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True
            )
        ])

    def __len__(self):
        return len(self.entity_list)

    def __getitem__(self, index):
        if (self.split == "train"):
            __rng = np.random
        else:
            __rng = np.random.RandomState(1019 + index)

        offset_unit = int(self.duration * self.offset_ratio)
        shift_unit = (self.duration - offset_unit * 2) / (self.t_res - 1)

        retries = -1
        while (retries < 3):
            retries += 1
            try:
                ent_name = self.entity_list[index]

                # load meta
                with open(os.path.join(self.major_path, f"{ent_name}.json")) as f:
                    ent_meta = json.load(f)

                if (ent_meta["duration"] < self.duration):
                    raise Exception(f"entity duration is less than { self.duration} seconds")

                # load analysis
                with open(os.path.join(self.analy_path, f"{ent_name}.json")) as f:
                    analy_data = json.load(f)

                segments = analy_data["segments"]
                segments = concat_segments(segments)

                if len(segments) < 3:
                    raise Exception("no sufficient segment after removing head/tail.")

                segments = segments[1:-1]

                clip_beg = None
                clip_end = None
                label = None
                rnd = __rng.rand()
                if (rnd < (1 / self.t_res)):
                    # determine transition segment
                    segment_idx = [i for i in range(len(segments))]

                    __rng.shuffle(segment_idx)

                    for s_idx in segment_idx:
                        s_beg = segments[s_idx]["start"]
                        s_end = segments[s_idx]["end"]

                        if (s_end - s_beg < (self.duration + 2 * offset_unit)):
                            continue
                        clip_beg = s_beg + offset_unit + __rng.randint(0, int(s_end - s_beg - self.duration - 2 * offset_unit) + 1)
                        clip_end = clip_beg + self.duration
                        s_mid = (clip_beg + clip_end) / 2
                        label = self.t_res - 1
                        break
                else:
                    # determine transition segment
                    segment_idx = [i + 1 for i in range(len(segments) - 1)]
                    label_idx = [i for i in range(self.t_res - 1)]

                    __rng.shuffle(segment_idx)
                    __rng.shuffle(label_idx)

                    for s_idx in segment_idx:
                        s_beg = segments[s_idx - 1]["start"]
                        s_end = segments[s_idx]["end"]
                        s_mid = segments[s_idx]["start"]

                        if (s_end - s_beg < (self.duration + 2 * offset_unit)):
                            continue

                        # select label
                        for l_idx in label_idx:
                            # determine clip interval
                            clip_beg = s_mid - (offset_unit + shift_unit * ((l_idx + 1) - 0.5))
                            clip_end = clip_beg + self.duration
                            # check clip within segment interval
                            if (clip_beg < s_beg) or (clip_end > s_end):
                                continue
                            else:
                                label = l_idx
                                break
                        else:
                            continue

                        break

                if (label is None):
                    raise Exception("no valid segment found")

                # load frames
                video_path = os.path.join(self.major_path, f"{ent_name}.mp4")
                if self.meta_only:
                    frames = None
                else:
                    frames = fetch_frames(video_path, clip_end - clip_beg, clip_beg, self.frames, self.img_preprocess)
                # frames = None
                if (self.with_audio):
                    audio_path = os.path.join(self.major_path, f"{ent_name}.mp3")
                    audio, _ = librosa.load(audio_path, sr=44100, offset=clip_beg, duration=self.duration)
                else:
                    audio = torch.tensor([])

                if (self.debug):
                    print("origin segments:")
                    for i, seg in enumerate(analy_data["segments"]):
                        print(i, seg)
                    print("process segments:")
                    for i, seg in enumerate(segments):
                        print(i, seg)
                    print("Segment : ({}) {:.4f}-{:.4f}-{:.4f}".format(s_idx, s_beg, s_mid, s_end))
                    print("Label:", label)
                    print("Clip: {:.4f}-{:.4f}-{:.4f}".format(clip_beg, s_mid, clip_end))
                    print("Path: ", video_path)

                    if (self.with_audio):
                        print(audio_path)

                return dict(
                    frames=frames,
                    labels=torch.tensor(label),
                    audio=audio,
                    idx=index,
                    timestamps=torch.tensor([clip_beg, s_mid, clip_end]),
                    video_path=video_path
                )

            except Exception as e:
                print(e, index)
                index = __rng.randint(0, len(self))

        raise NotImplementedError()


class RegressMusicVideoDataset(Dataset):
    def __init__(
        self,
        split,
        frames=16,
        duration=30,
        px=336,
        portion=1.0,
        offset_ratio=0.2,
        reverse=False,
        with_audio=False,
        debug=False,
        dts_ver="v1"
    ):
        super().__init__()
        self.split = split
        self.frames = frames
        self.major_path = f"/scratch4/users/od/YTCharts/{dts_ver}/{split}/"
        self.analy_path = f"/scratch4/users/od/YTCharts/analysis/"
        self.duration = duration
        self.with_audio = with_audio
        self.debug = debug
        self.offset_ratio = offset_ratio
        self.entity_list = sorted(
            [
                os.path.basename(file)[:-4]
                for file in glob(f"{self.major_path}/*.mp3")
            ],
            reverse=reverse
        )
        self.entity_list = self.entity_list[:int(len(self.entity_list) * portion)]
        self.img_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (px, px),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True
            )
        ])

    def __len__(self):
        return len(self.entity_list)

    def __getitem__(self, index):
        if (self.split == "train"):
            __rng = np.random
        else:
            __rng = np.random.RandomState(1019 + index)

        offset_unit = int(self.duration * self.offset_ratio)

        retries = -1
        while (retries < 3):
            retries += 1
            try:
                ent_name = self.entity_list[index]

                # load meta
                with open(os.path.join(self.major_path, f"{ent_name}.json")) as f:
                    ent_meta = json.load(f)

                if (ent_meta["duration"] < self.duration):
                    raise Exception(f"entity duration is less than { self.duration} seconds")

                # load analysis
                with open(os.path.join(self.analy_path, f"{ent_name}.json")) as f:
                    analy_data = json.load(f)

                segments = deepcopy(analy_data["segments"])
                segments = concat_segments(segments)

                if len(segments) < 3:
                    raise Exception("no sufficient segment after removing head/tail.")

                segments = segments[1:-1]

                clip_beg = None
                clip_end = None
                label = None
                rnd = __rng.rand()
                if (rnd < 0.3):
                    # determine transition segment
                    segment_idx = [i for i in range(len(segments))]

                    __rng.shuffle(segment_idx)

                    for s_idx in segment_idx:
                        s_beg = segments[s_idx]["start"]
                        s_end = segments[s_idx]["end"]

                        if (s_end - s_beg < (self.duration + 2 * offset_unit)):
                            continue
                        clip_beg = s_beg + offset_unit + __rng.randint(0, int(s_end - s_beg - self.duration - 2 * offset_unit) + 1)
                        clip_end = clip_beg + self.duration
                        s_mid = (clip_beg + clip_end) / 2
                        norm = -1
                        label = 0
                        break
                else:
                    # determine transition segment
                    segment_idx = [i + 1 for i in range(len(segments) - 1)]

                    __rng.shuffle(segment_idx)

                    for s_idx in segment_idx:
                        s_beg = segments[s_idx - 1]["start"]
                        s_end = segments[s_idx]["end"]
                        s_mid = segments[s_idx]["start"]

                        if (s_end - s_beg < (self.duration + 2 * offset_unit)):
                            continue

                        left_bound = max(s_mid - self.duration, s_beg) + offset_unit
                        right_bound = min(s_mid + self.duration, s_end) - offset_unit
                        clip_beg = __rng.random() * (right_bound - left_bound - self.duration) + left_bound
                        clip_end = clip_beg + self.duration
                        norm = (s_mid - clip_beg) / self.duration
                        label = 1
                        break

                if (label is None):
                    raise Exception("no valid segment found")

                # load frames
                video_path = os.path.join(self.major_path, f"{ent_name}.mp4")
                frames = fetch_frames(video_path, clip_end - clip_beg, clip_beg, self.frames, self.img_preprocess)
                # frames = None
                if (self.with_audio):
                    audio_path = os.path.join(self.major_path, f"{ent_name}.mp3")
                    audio, _ = librosa.load(audio_path, sr=44100, offset=clip_beg, duration=self.duration)
                else:
                    audio = torch.tensor([])

                if (self.debug):
                    print("Origin Segments:")
                    for i, seg in enumerate(analy_data["segments"]):
                        print(i, seg)
                    print("Process Segments:")
                    for i, seg in enumerate(segments):
                        print(i, seg)
                    print("Segment: ({}) {:.4f}-{:.4f}-{:.4f}".format(s_idx, s_beg, s_mid, s_end))
                    print("Label:", label)
                    print("Clip: {:.4f}-{:.4f}-{:.4f}".format(clip_beg, s_mid, clip_end))
                    print("Path:", video_path)
                    print("Norm:", norm)

                    if (self.with_audio):
                        print(audio_path)

                return dict(
                    frames=frames,
                    labels=torch.tensor(label),
                    norms=torch.tensor(norm),
                    audio=audio,
                    idx=index,
                    timestamps=torch.tensor([clip_beg, s_mid, clip_end]),
                    video_path=video_path
                )

            except Exception as e:
                print(e, index)
                index = __rng.randint(0, len(self))

        raise NotImplementedError()


class FullVideoDataset(Dataset):

    def __init__(self, frames=32, px=336, portion=1.0):
        super().__init__()
        self.frames = frames
        self.video_folder = "/scratch4/users/od/YTCharts/videos/"
        self.meta_folder = self.video_folder.replace("videos", "metas")

        self.entity_list = sorted(
            [
                os.path.basename(file)[:-4]
                for file in glob(f"{self.video_folder}/*.mp4")
            ]
        )

        self.entity_list = self.entity_list[:int(len(self.entity_list) * portion)]

        self.img_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (px, px),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True
            )
        ])

    def __len__(self):
        return len(self.entity_list)

    def __getitem__(self, index):
        try:
            ent_name = self.entity_list[index]
            video_path = os.path.join(self.video_folder, f"{ent_name}.mp4")
            meta_path = os.path.join(self.meta_folder, f"{ent_name}.json")

            # load meta
            with open(meta_path) as f:
                ent_meta = json.load(f)

            # load frames
            frames = fetch_frames(video_path, 60, ent_meta["duration"] // 2 - 30, self.frames, self.img_preprocess)

            return dict(
                frames=frames,
                video_path=video_path,
                meta_path=meta_path
            )

        except Exception as e:
            print(e, index)
            return self.__getitem__(np.random.randint(0, len(self)))


if __name__ == "__main__":
    import torch
    import random
    import numpy as np

    def seed_everything(seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    loader = DataLoader(MusicVideoDataset("test", duration=10, frames=10), batch_size=4, num_workers=16, shuffle=True, pin_memory=False)
    for i in tqdm(loader):
        pass
    # for i in tqdm(range(len(dts))):
    #     dts[i]
