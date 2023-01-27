import numpy as np
import cv2
import os
from typing import List

import torch
from torchcontentarea import estimate_area_learned
import pandas as pd


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


class ProcessVideos():
    def __init__(self, video_files: List[str]) -> None:
        self.captures = [cv2.VideoCapture(video_file) for video_file in video_files]

    def run(self):
        for capture in self.captures:
            while True:
                ret, frame = capture.read()
                if ret:
                    self.process(frame)
                else:
                    break
            capture.release()


    def process(self, frame: np.ndarray) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.
        
        area = estimate_area_learned(frame)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)
