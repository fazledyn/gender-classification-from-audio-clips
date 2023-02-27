import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa



def make_mfcc(filename):

    y, sr = librosa.load(f"./dls-train-clean/{filename}")
    yt , index = librosa.effects.trim(y=y)
    length = librosa.get_duration(y=yt)

    #   pre computed log power mel spectrogram
    # log_S = librosa.power_to_db(S, ref=np.max)    

    S = librosa.feature.mfcc(y=y, sr=sr)

    librosa.display.specshow(S, sr=sr)
    filename = filename.replace(".mp3", ".png")

    plt.savefig(f"./mfcc/{filename}", transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()

    return filename, length


def main():

    #   reading csv
    df = pd.read_csv("./dls-train-clean.csv")
    print(df.head())
    print(df.shape)

    #   only taking audio length > 2
    df = df.loc[ df["length"] > 2 ]
    print(df.shape)

    #   adding new column imgname
    df["imgname"] = ""
    print(df.head())
    print("Generating MFCC now ...")

    #   iteration steps
    for index, row in df.iterrows():

        print(f"At {index}/{len(df)}", end="\r")
        imgname, length = make_mfcc(row["filename"])

        df.at[index, "imgname"] = imgname
        df.at[index, "length"] = length

    print("Complete!")
    df.to_csv("mfcc_dls-train-clean.csv", index=False)


if __name__ == "__main__":
    main()
