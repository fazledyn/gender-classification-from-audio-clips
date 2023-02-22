"""
##############################################################
##############################################################
PROCESS MCV DATA to HAVE EQUAL # OF GENDER
"""

# import pandas as pd

# IN_CSV  = "./cv-dev-merged.csv"
# OUT_CSV = "./cv-dev-balanced.csv"

# df = pd.read_csv(IN_CSV)

# df_feml = df.loc[ df["gender"] == "female" ]

# df_male = df.loc[ df["gender"] == "male"   ]
# df_male = df_male.sample(n=len(df_feml), random_state=66)

# df_new = pd.concat([df_male, df_feml], ignore_index=True)

# print("Length of new df:", len(df_new))
# print("Length of Female:", len(df_feml))
# print("Length of Male:", len(df_male))

# df_new.to_csv(OUT_CSV, index=False)

"""
##############################################################
##############################################################
CREATING SPECTROGRAM AND FINDING OUT LENGTH, LABELS ETC.
"""

import matplotlib.pyplot as plt
import librosa, librosa.display
import pandas as pd
import numpy as np
import os


def make_spec(filename):

    y, sr = librosa.load(f"./cv-dev-merged/{filename}")
    yt , index = librosa.effects.trim(y=y)
    length = librosa.get_duration(y=yt)

    S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)    

    librosa.display.specshow(log_S, sr=sr)
    filename = filename.replace(".mp3", ".png")

    plt.savefig(f"./cv-dev-balanced/{filename}", transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()
    return filename, length


IN_CSV = "./cv-dev-balanced.csv"
OUT_CSV = "./cv-dev-balanced-new.csv"

df = pd.read_csv(IN_CSV)
df["imgname"] = 0
df["length"] = 0
df["label"] = -1

for index, row in df.iterrows():

    print(f"At {index}/{len(df)}", end="\r")
    imgname, length = make_spec(row["filename"])
    df.at[index, "imgname"] = imgname
    df.at[index, "length"] = length

    if row["gender"] == "male":
        df.at[index, "label"] = 1
    elif row["gender"] == "female":
        df.at[index, "label"] = 0

print("")
print("COMPLETE!")

df.to_csv(OUT_CSV)


"""
##############################################################
##############################################################
"""