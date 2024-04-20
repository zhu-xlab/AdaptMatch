## Datasets
First, please organize your datasets according to the following structure with WHU_Roads as an example:
```plaintext
./datasets/WHU_Roads/
│
├── images/                  
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
│   │
├── labels/          
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
```
Here, in  'images/' and 'labels/', all the images and labels needs to be cropped to 512x512. In 'lists', 'train_percent%_labeled.txt' and 'train_percent%_unlabeled.txt' contain the paths of labeled and unlabeled image-label pairs.


# Pretrained Weights
The used SegFormer's codes and pretrained weights are from https://github.com/bubbliiiing/segformer-pytorch. Please download the weights to the directory "./checkpoints/".

## Training
You can train a model (with WHU_Roads as an example) as: 
```plaintext
python train.py -g 0 -m Adaptmatch --model SegFormer -d WHU_Roads -p 1 --num-iters 20000
```


## Test
After training, the model can be tested as:
```plaintext
python test.py --gpu 0 --method Adaptmatch --model SegFormer --dataset WHU_Roads --percent 1
```
