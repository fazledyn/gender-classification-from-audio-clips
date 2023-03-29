from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import sys


IN_FILE  = "./csv/cv-dev-cleaned.csv" 
OUT_FILE = "./csv/cv-dev-cleaned-spec.csv"


def get_img_name(filename):
    return filename.replace(".mp3", ".png")


def get_audio_length(filename):
    y, sr = librosa.load(f"../dataset/cv-dev/cv-dev-cleaned/{filename}")
    yt , index = librosa.effects.trim(y=y)
    return librosa.get_duration(y=yt, sr=sr)


def generate_spec(filename):
    y, sr = librosa.load(f"../dataset/cv-dev/cv-dev-cleaned/{filename}")
    yt , index = librosa.effects.trim(y=y)

    S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)

    # librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    librosa.display.specshow(log_S, sr=sr)

    plt.savefig(f"../dataset/cv-dev/spec/{get_img_name(filename)}", transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()


def main():

    try:
        os.mkdir("../dataset/cv-dev/spec/")
        os.mkdir("../dataset/cv-dev/spec/cv-valid-dev")
        os.mkdir("../dataset/cv-dev/spec/cv-other-dev")
    except:
        pass

    df_in = pd.read_csv(IN_FILE)
    df_out = df_in.copy()
    df_out["img"] = ""
    df_out["length"] = 0

    print("Iteration starts...")
    len_df = len(df_out)

    for index, row in df_out.iterrows():

        print(f"Processing {index}/{len_df}...", end="\r")
        audio_len = get_audio_length(row["filename"])

        if audio_len > 2:
            row["img"] = get_img_name(row["filename"])
            row["length"] = audio_len
            # generate_spec(row["filename"])
            # print(f"Generating spec for {row['filename']}...")
        else:
            df_out.drop(index, inplace=True)
            # print(f"Removing {row['filename']} from dataset...")

    df_out.to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    main()