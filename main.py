import os
from utils import recursive_scan2df, ProcessVideos


def main() -> None:
    folder = "/nfs/home/mhuber/data/endoscopic_data/heichole/Videos/Full_SD"
    videos_df = recursive_scan2df(folder, ".mp4")
    videos = [os.path.join(folder, row.folder, row.file) for _, row in videos_df.iterrows()]
    print(videos)

    process_videos = ProcessVideos(videos, "/nfs/home/mhuber/data/endoscopic_data/heichole_single_frames_cropped_new")
    process_videos.run(processes=1)

if __name__ == "__main__":
    main()
