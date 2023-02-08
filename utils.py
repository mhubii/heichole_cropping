import os
import pathlib
from multiprocessing import Pool
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchcontentarea import estimate_area_learned

from crop import crop_area


def recursive_scan2df(folder: str, postfix: str = ".jpg") -> pd.DataFrame:
    # scan folder for images and return dataframe
    df = pd.DataFrame(columns=["folder", "file"])

    print(f"Scanning {folder} recursively for {postfix} files")
    for root, subdirs, files in os.walk(folder):
        files = [x for x in files if postfix in x]
        if files:
            dic_list = [
                {"folder": root.replace(folder, "").strip("/"), "file": x}
                for x in files
            ]
            df = df.append(dic_list, ignore_index=True)
    return df


def unique_video_train_test(
    df: pd.DataFrame,
    train_split: float = 0.8,
    tolerance: float = 0.05,
    random_state: int = 42,
):
    r"""Splits videos into train and test set.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'folder': , 'file': , 'vid': , 'frame': }
        train_split (float): Fraction of train data, default 0.8
        tolerance (float): Split tolerance, train_split + tolerance <= len(df[df.train] == False)/len(df) <= train_split + tolerance
        random_state (int): Random state for deterministic splitting

    Return:
        df (pd.DataFrame): Pandas dataframe, contain {'folder': , 'file': , 'vid': , 'frame': , 'train': }
    """
    # find unique videos
    unique_vid = df.vid.unique()

    _, test_vid = train_test_split(
        unique_vid, train_size=train_split, random_state=random_state
    )

    df["train"] = True
    df.loc[df.vid.isin(test_vid), "train"] = False

    # assert if fraction off
    fraction = len(df[df.train == False]) / len(df)
    assert np.isclose(
        fraction, 1 - train_split, atol=tolerance
    ), "Train set fraction {:.3f} not close enough to (1 - train_split) {} at tolerance {}".format(
        fraction, 1 - train_split, tolerance
    )

    return df


def generate_path(prefix: str) -> None:
    if not os.path.exists(prefix):
        print(f"Generating path at {prefix}")
        pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)


class ProcessVideos:
    def __init__(
        self,
        video_files: List[str],
        output_path: str,
        shape: Tuple[int] = (320, 240),
        aspect_ratio: float = 4.0 / 3.0,
        device="cuda",
    ) -> None:
        self.video_files = video_files
        self.video_names = [
            video_file.split("/")[-1].split(".")[0] for video_file in self.video_files
        ]
        self.captures_splits = [0] * len(self.video_files)
        self.output_path = output_path
        self.shape = shape
        self.aspect_ratio = aspect_ratio
        self.device = device

    def run(self, processes: int = 1):
        process_pool = Pool(processes)
        capture_idcs = [capture_idx for capture_idx in range(len(self.video_files))]
        process_pool.map(self.process_video, capture_idcs)

    def process_video(self, capture_idx: int):
        print(f"Start to process video {self.video_names[capture_idx]}...")
        capture = cv2.VideoCapture(self.video_files[capture_idx])
        previous_success = False
        frame_idx = 0
        output_path = None
        while True:
            ret, frame = capture.read()
            if ret:
                processed_img, success = self.process(frame)

                if previous_success == False and success == True:
                    output_path = os.path.join(
                        self.output_path,
                        self.video_names[capture_idx],
                        f"split_{self.captures_splits[capture_idx]}",
                    )
                    generate_path(output_path)
                    self.captures_splits[capture_idx] += 1
                    frame_idx = 0
                if success:
                    np.save(
                        os.path.join(output_path, f"frame_{frame_idx}.npy"),
                        processed_img,
                    )
                    frame_idx += 1
                previous_success = success
            else:
                break
        capture.release()
        print(f"Done processing video {self.video_names[capture_idx]}.")

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if (frame - 255).count_nonzero().item() == 0:
            # video censored
            return frame.squeeze().permute(1, 2, 0).cpu().numpy(), False

        area = estimate_area_learned(frame)
        if area.count_nonzero().item() == 0:
            # no circle found (return entire frame)
            frame = frame.squeeze().permute(1, 2, 0).cpu().numpy()
            frame = cv2.resize(frame, self.shape)
            return frame, True

        cropped_frame = crop_area(area.squeeze(), frame, aspect_ratio=self.aspect_ratio)
        print(area)
        print(cropped_frame.shape)

        cropped_frame = cropped_frame.squeeze().permute(1, 2, 0).cpu().numpy()
        cropped_frame = cv2.resize(cropped_frame, self.shape)

        return cropped_frame, True
