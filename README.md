## Datasets

Please organize your datasets according to the following structure:

```plaintext
./datasets/
│
├── images/                   # 训练集文件夹
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
│   │
├── labels/           # 类别2的图像
│   ├── label_001.png
│   ├── label_002.png
│   └── ...
│
└── lists
│   ├── train_1%_labeled.txt
│   ├── train_1%_unlabeled.txt
│   ├── train_5%_labeled.txt
│   ├── train_5%_unlabeled.txt
│   ├── train_20%_labeled.txt
│   ├── train_20%_unlabeled.txt
│   ├── val.txt
│   ├── test.txt
