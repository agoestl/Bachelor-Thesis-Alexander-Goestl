## Gains from Machine-Learning-Based Covariate Adjustment in Regression Discontinuity Designs: Evidence from Financial Data

This repository contains materials, notes, and progress related to my bachelor thesis.

## Contact

### Me
**Alexander Goestl**
E-Mail: goestl.alexander@gmail.com
GitHub: [agoestl](https://github.com/agoestl)

### Supervisor
**Dr. Tomasz Olma**
E-Mail: t.olma@lmu.de
GitHub: [tomaszolma](https://github.com/tomaszolma)
Website: https://tomaszolma.github.io/

## Download Data

All data files are available here: https://drive.google.com/drive/folders/1wr2OrU9V5gKGlB60zQ2NVvmTP1-KicTH?usp=drive_link

Move the downloaded data folder in each corresponding *_Replication folder of this repository. 

## Folder Structure

The repository is organized into three replication folders and one setup folder. Each replication folder contains the scripts, generated results, figures, and the corresponding reference paper. The data folders are not included in the repository and have to be added manually after downloading the data.

```text
Bachelor-Thesis-Alexander-Goestl/
├── README.md                              # General overview and reproduction instructions
├── setup/                                 # Environment and version information
│   ├── requirements.txt                   # Required Python packages
│   └── version.txt                        # Software/package versions used
│
├── Barber_Replication/                    # Barber, Huang, and Odean replication
│   ├── guide.md                           # Specific instructions for this replication
│   ├── Barber_Huang_Odean_2022.pdf        # Original paper used for the replication
│   ├── scripts/                           # Scripts for data preparation and analysis
│   ├── results/                           # Generated tables and numerical outputs
│   ├── figures/                           # Generated figures
│   └── data/                              # Data folder added manually after download
│
├── Griffin_Replication/                   # Griffin and Shams replication
│   ├── guide.md                           # Specific instructions for this replication
│   ├── Griffin_Shams_2020_WP.pdf          # Original paper used for the replication
│   ├── scripts/                           # Scripts for data preparation and analysis
│   ├── results/                           # Generated tables and numerical outputs
│   ├── figures/                           # Generated figures
│   └── data/                              # Data folder added manually after download
│
└── Ouazad_Replication/                    # Ouazad and Kahn replication
    ├── guide.md                           # Specific instructions for this replication
    ├── Ouazad_Kahn_2023.pdf               # Original paper used for the replication
    ├── scripts/                           # Scripts for data preparation and analysis
    ├── results/                           # Generated tables and numerical outputs
    ├── figures/                           # Generated figures
    └── data/                              # Data folder added manually after download
```

## Meeting Overview

### 24.03 — Initial Meeting

- get to know each other in person
- check for misunderstandings
- specify the scope of replication to the finance sector
- lookup datasources
- discussed the timeline

### 09.04 — Meeting

- presentation of replication results from griffin and barber
- further gathering of possible replication papers is planned
- discussed the possible research question
- preparation of "Registration of Thesis"

### 17.04 — Meeting

- presentation of replication results from griffin and barber
- presentation of possible additional replication paper
- decided for Research Questions
- decided for Name of the Thesis
- decided for Structure of Thesis
- preparing "Registration of Thesis" paper

### 30.04 — Meeting

- regarding Griffin no effect covariates
- regarding Barber replication polynomial replication
- widen model bandwidth
- interactions between covariates (for LASSO)
- citation guidlines
- dates for handing in the paper and defense

### 21.05 - Meeting

- discussion about subsection selection
- check the replication rdd assumptions
- discussion about simulation
- use/implementation of robustness/falsification tests
- implementation of rdbust to compare against linear adjustments
- removing adaptive-lasso
- working with diffrent seeds
- further work

### 11.06 - Meeting

- discussion about the first 3 Chapters
- AI Usage Chapter
- discussion about a retracted source
- structure figures and tables
- confirming handing in the paper
- planing the date for defense (06.07.2026)