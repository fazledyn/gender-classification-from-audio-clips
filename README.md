![Project Cover Image](/docs/cover.svg)

# Overview
In this project, we aim to build a machine learning model that can identify the gender of a person from their voice recording. In the process, we use two intermediary data representation format of the audio clips- **Mel Spectrogram** (Mel) and **Mel-Frequency Cepstral Coefficients** (MFCC).


# Datasets
**[MCV]** Common Voice by Mozilla.org (https://www.kaggle.com/datasets/mozillaorg/common-voice)

**[DLS]** Bengali Common Voice Speech Dataset (https://www.kaggle.com/competitions/dlsprint)


# Proposed Solution
## Mel-Frequency Cepstral Coefficients (MFCC)
![Project Cover Image](/docs/mfcc.png)
## Mel Spectrogram
![Project Cover Image](/docs/mel.png)


# Notebook Details
## Training
The `training` folder contains four notebooks. Each of the notebooks are named as: `[Data-Type]_[Dataset]_[Model]`. These notebooks are used to train individual models on the train datasets.

```      
└── training
    ├── mel_dls_resnet50_train.ipynb 
    ├── mel_mcv_resnet50.ipynb       
    ├── mfcc_dls_train_resnet50.ipynb
    └── mfcc_mcv_resnet50.ipynb
```

## Evaluation
The `evaluation` folder contains four notebooks. Each of the notebooks are named as: `[Data-Type]_[Datase#1]_on_[Dataset#2]`. The models trained on `Dataset#1` are used to evaluate `Dataset#2`.

```
└── evaluation
    ├── mel_dls_on_mcv.ipynb
    ├── mel_mcv_on_dls.ipynb
    ├── mfcc_dls_on_mcv.ipynb        
    └── mfcc_mcv_on_dls.ipynb
```

In the report mentioned in the [presentation](#presentation-report), the comparison between models are shown.


# Model Details
- Architecture: ResNet50
- Learning Rate: 0.0001
- Adam Optimizer


# Presentation Report
https://docs.google.com/presentation/d/14BWOq6YSmO3GqZHEvCou43Z5A4dlOmKq4pjqUqgZALU/


# References
[1]  **Speaker Gender Recognition Based on Deep Neural Networks and ResNet50** (https://doi.org/10.1155/2022/4444388)

[2]  **A Machine Learning Approach to Automating Bengali Voice Based Gender Classification** (https://ieeexplore.ieee.org/document/9117385)
