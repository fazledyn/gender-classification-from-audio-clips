"""
###############################################
###############################################
###############################################

Sampling Equal # of male/female from dataset
"""

# import pandas as pd

# IN_CSV  = "./dls-train-clean.csv"
# OUT_CSV = "./dls-train-balanced.csv"

# df = pd.read_csv(IN_CSV)

# df_feml = df.loc[ df["gender"] == "female" ]

# df_male = df.loc[ df["gender"] == "male"   ]
# df_male = df_male.sample(n=len(df_feml), random_state=66)

# df_new = pd.concat([df_male, df_feml], ignore_index=True)

# print("Length of new df:", len(df_new))
# df_new.to_csv(OUT_CSV, index=False)

"""
###############################################
###############################################
###############################################

Deleting train_files that aren't labelled
"""

import matplotlib.pyplot as plt
import librosa, librosa.display
import pandas as pd
import numpy as np
import os


def make_spec(filename):

    y, sr = librosa.load(f"./dls-train-clean/{filename}")
    yt , index = librosa.effects.trim(y=y)

    S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(log_S, sr=sr)
    filename = filename.replace(".mp3", ".png")

    plt.savefig(f"./dls-train-balanced/{filename}", transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()
    return filename


IN_CSV = "./dls-train-balanced.csv"
OUT_CSV = "./dls-train-balanced-new.csv"

df = pd.read_csv(IN_CSV)
df["imgname"] = 0

for index, row in df.iterrows():
    print(f"At {index}/{len(df)}", end="\r")
    imgname = make_spec(row["filename"])
    df.at[index, "imgname"] = imgname

print("")
print("COMPLETE!")

df.to_csv(OUT_CSV)


"""
###############################################
###############################################
###############################################

Extracting filename and gender from the CSV
"""

# import librosa
# import pandas as pd

# IN_CSV = "./dls-train-clean.csv"
# OUT_CSV = "./dls-train-clean-length.csv"

# df = pd.read_csv(IN_CSV)
# df_out = df.copy()
# lim = len(df)

# df["length"] = 0
# df["label"] = -1


# def get_length(filename):
#     y, sr = librosa.load(f"./dls-train-clean/{filename}")
#     yt, sr = librosa.effects.trim(y=y)
#     return librosa.get_duration(y=yt)


# def get_gender(gender):
#     return int(gender == "male")


# # df["length"] = df["filename"].apply(lambda x: get_length(x))

# for index, row in df.iterrows():
#     df.at[index, "length"] = get_length(row["filename"])
#     df.at[index, "label"]  = get_gender(row["gender"])

#     print(f"At {index}/{lim}", end="\r")
#     # print("Head")
#     # print(df.head)


# print("")
# print("COMPLETE!")

# df.to_csv(OUT_CSV)

"""
###############################################
###############################################
###############################################

Creating Mel-Spectrogram from the files
"""

# import matplotlib.pyplot as plt
# import librosa, librosa.display
# import pandas as pd
# import numpy as np


# def make_spec(filename):

#     y, sr = librosa.load(f"./dls-train-clean/{filename}")
#     yt , index = librosa.effects.trim(y=y)

#     S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=128)
#     log_S = librosa.power_to_db(S, ref=np.max)

#     librosa.display.specshow(log_S, sr=sr)
#     filename = filename.replace(".mp3", ".png")

#     plt.savefig(f"./spec/{filename}", transparent=True, pad_inches=0, bbox_inches='tight')
#     plt.close()
#     return filename


# IN_CSV = "./dls-train-clean.csv"
# df = pd.read_csv(IN_CSV)
# df_sub = df.copy()

# df_sub["imgname"] = 0

# lim = len(df)

# for index, row in df.iterrows():
#     df_sub.at[index, "imgname"] = make_spec(row["filename"])
#     print(f"At {index}/{lim}", end="\r")

#     if index == 8000:
#         print("\n\nBreaking at 8000")
#         break

# print("")
# print("COMPLETE!")

# df_sub.to_csv("./dls-clean-sub.csv", index=False)

"""
###############################################
###############################################
###############################################
"""
