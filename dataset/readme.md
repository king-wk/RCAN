# Dataset

这是存储数据集的文件，文件结构如下：
```
dataset
├── DIV2K  # 1-800 for train, 801-900 for test
│   ├── DIV2K_train_LR_bicubic
│   │   ├── X2
│   │   │   ├── 0001x2.png
│   │   │   ├── ...
│   │   │   └── 0900x2.png
│   │   ├── X3
│   │   │   ├── 0001x3.png
│   │   │   ├── ...
│   │   │   └── 0900x3.png
│   │   └── X4
│   │   │   ├── 0001x4.png
│   │   │   ├── ...
│   │   │   └── 0900x4.png
│   └── DIV2K_train_HR
│       ├── 0001.png
│       ├── ...
│       └── 0900.png
└── benchmark  # for test
    ├── Set5
    │   ├── LR_bicubic
    │   │   ├── X2
    │   │   │   ├── baboonx2.png
    │   │   │   └── ...
    │   │   ├── X3
    │   │   │   ├── baboonx3.png
    │   │   │   └── ...
    │   │   └── X4
    │   │       ├── baboonx4.png
    │   │       └── ...
    │   └── HR
    │       ├── baboon.png
    │       └── ...
    ├── Set14
    │   ├── LR_bicubic
    │   │   ├── X2
    │   │   │   └── ...
    │   │   ├── X3
    │   │   │   └── ...
    │   │   └── X4
    │   │       └── ...
    │   └── HR
    │       └── ...
    ├── B100
    │   ├── LR_bicubic
    │   │   ├── X2
    │   │   │   └── ...
    │   │   ├── X3
    │   │   │   └── ...
    │   │   └── X4
    │   │       └── ...
    │   └── HR
    │       └── ...
    └── Urban100
        ├── LR_bicubic
        │   ├── X2
        │   │   └── ...
        │   ├── X3
        │   │   └── ...
        │   └── X4
        │       └── ...
        └── HR
            └── ...
```
