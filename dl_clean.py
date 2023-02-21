"""
###############################################
###############################################
###############################################

Extracting filename and gender from the CSV
"""

import librosa
import pandas as pd

IN_CSV = "./dls-train-clean.csv"
OUT_CSV = "./dls-train-clean-length.csv"

df = pd.read_csv(IN_CSV)
df_out = df.copy()
lim = len(df)

df["length"] = 0
df["label"] = -1


def get_length(filename):
    y, sr = librosa.load(f"./dls-train-clean/{filename}")
    yt, sr = librosa.effects.trim(y=y)
    return librosa.get_duration(y=yt)


def get_gender(gender):
    return int(gender == "male")


# df["length"] = df["filename"].apply(lambda x: get_length(x))

for index, row in df.iterrows():
    df.at[index, "length"] = get_length(row["filename"])
    df.at[index, "label"]  = get_gender(row["gender"])

    print(f"At {index}/{lim}", end="\r")
    # print("Head")
    # print(df.head)


print("")
print("COMPLETE!")

df.to_csv(OUT_CSV)

"""
###############################################
###############################################
###############################################
"""
