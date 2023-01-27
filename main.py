from utils import recursive_scan2df


def main() -> None:
    folder = "/nfs/home/mhuber/data/endoscopic_data/heichole/Videos/Full_SD"
    print(recursive_scan2df(folder, ".mp4"))


if __name__ == "__main__":
    main()
