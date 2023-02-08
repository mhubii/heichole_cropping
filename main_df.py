import os

from utils import recursive_scan2df, unique_video_train_test


def main() -> None:
    folder = "/nfs/home/mhuber/data/endoscopic_data/heichole_single_frames_cropped"
    frames_df = recursive_scan2df(folder, ".npy")
    print(frames_df)
    frames_df.to_csv(os.path.join(folder, "log.csv"))
    frames_df.to_pickle(os.path.join(folder, "log.pkl"))

    test_train_frames_df = unique_video_train_test(frames_df)
    test_train_frames_df.to_csv(os.path.join(folder, "log_test_train.csv"))
    test_train_frames_df.to_pickle(os.path.join(folder, "log_test_train.pkl"))


if __name__ == "__main__":
    main()
