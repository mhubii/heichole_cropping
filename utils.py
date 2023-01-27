# import torchcontentarea
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
