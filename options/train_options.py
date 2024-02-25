import argparse


class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser( description="training script for FDA" )        
        # gpu
        parser.add_argument('-g', "--gpu", type=int, default=0, help="set GPU.")

        # dataset
        parser.add_argument("--num-classes", type=int, default=1, help="Number of classes for ISPRS.")
        parser.add_argument("--data-dir", type=str, default='/data/wei/Datasets/RSSeg/dataset')       
        parser.add_argument('-d', "--dataset", type=str, default='INRIA/Austin', help="percent of training set.")
                                # choices=['INRIA/Austin', 'INRIA/Chicago', 'INRIA/Vienna', 'INRIA/Kitsap', 'INRIA/Tyrol', 
                                # 'DeepGlobe', 'LRSNY', 'WHU_Roads', 'Ottawa']         
        parser.add_argument('-p', "--percent", type=int, default=100, help="percent of training set.")
        parser.add_argument("--train-lbl-list", type=str, default='lists/train_percent%_labeled.txt')
        parser.add_argument("--train-unl-list", type=str, default='lists/train_percent%_unlabeled.txt')
        parser.add_argument("--val-list", type=str, default='lists/val.txt')

        # model
        parser.add_argument("--model", type=str, default='SegFormer', 
                                choices=['DeepLab_V3plus', 'HRNet', 'EfficientUNet', 'SegFormer'])
        parser.add_argument('-m', "--method", type=str, default='Adaptmatch', \
                                choices=['Sup', 'Fixmatch', 'Adaptmatch'])

        # optimization
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=4, help="input batch size.")
        parser.add_argument("--max-img-size", type=int, default=512, help="maximum image size.")
        parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")

        parser.add_argument("--restore", type=bool, default=False, help="restore checkpoint or not.")
        parser.add_argument("--save-dir", type=str, default='./checkpoints', help="Where to save snapshots of the model.")
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--print-freq", type=int, default=50)
        parser.add_argument("--eval-freq", type=int, default=500)
        parser.add_argument("--num-iters", type=int, default=10000)

        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)






