import os
from utils import recursive_scan2df, ProcessVideos


def main() -> None:
    folder = "/media/martin/Samsung_T5/data/endoscopic_data/10s_video"
    videos_df = recursive_scan2df(folder, ".mp4")
    videos = [os.path.join(folder, row.folder, row.file) for _, row in videos_df.iterrows()]
    print(videos)

    process_videos = ProcessVideos(videos)
    process_videos.run()



if __name__ == "__main__":
    main()
