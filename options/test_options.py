import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--gpu", type=int, default=0, help="set GPU.")
        parser.add_argument("--num_classes", type=int, default=1, help="Number of classes for ISPRS.")
        parser.add_argument("--percent", type=int, default=5, help="percent of training set.")
        parser.add_argument("--data_dir", type=str, default='/data/wei/Datasets/RSSeg')
        parser.add_argument("--dataset", type=str, default='Bijie')
                                # choices=['INRIA/Austin', 'Chicago', 'Vienna', 'Kitsap', 'Tyrol', 
                                # 'DeepGlobe', 'LRSNY', 'WHU_Roads', 'Ottawa', 
                                # 'Massachusetts_Buildings', 'Bijie']        
        parser.add_argument("--model", type=str, default='SegFormer', 
                                choices=['DeepLab_V3plus', 'HRNet', 'FCDenseNet67', 'EfficientUNet', \
                                         'Swin_Transformer', 'SegFormer', 'BuildFormer'])
        parser.add_argument("--method", type=str, default='Sup+Adapmatch', \
                                choices=['Sup', 'Sup+Fixmatch', 'Sup+Adapmatch', 'Sup+lblAdapmatch'])
        parser.add_argument("--save_dir", type=str, default='./checkpoints', help="Where to save tSNEs.")
        parser.add_argument("--batch_size", type=int, default=16, help="input batch size.")
        parser.add_argument("--num_workers", type=int, default=4, help="number of threads.")

        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

