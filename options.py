import argparse

class TrainOptions_E():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='', help='path of data')
            
        # test
        self.parser.add_argument('--ckpt_path', type=str, default="L2024.pth", help='Path to the pre-trained weights')
        self.parser.add_argument('--vi_path', type=str, default="/ceph_data/jhlee39/workspace/repos/ETRI/data/ETRI_Night/rgb_1", help='Path to the Visible images')
        self.parser.add_argument('--ir_path', type=str, default="/ceph_data/jhlee39/workspace/repos/ETRI/data/ETRI_Night/Thermal_1", help='Path to the Infrared images')
        self.parser.add_argument('--out_path', type=str, default="/ceph_data/jhlee39/workspace/repos/ETRI/data/ETRI_Night/fused_denoise_inference", help='Path to save the Fusion results')
            
    
    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt



class TrainOptions_K():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='', help='path of data') 
            
        # test
        self.parser.add_argument('--ckpt_path', type=str, default="L2024.pth", help='Path to the pre-trained weights')
        self.parser.add_argument('--vi_path', type=str, default="/ceph_data/jhlee39/workspace/repos/ETRI/data/kiro_night/RGB_enhanced", help='Path to the Visible images')
        self.parser.add_argument('--ir_path', type=str, default="/ceph_data/jhlee39/workspace/repos/ETRI/data/kiro_night/Thermal", help='Path to the Infrared images')
        self.parser.add_argument('--out_path', type=str, default="/ceph_data/jhlee39/workspace/repos/ETRI/data/kiro_night/fused", help='Path to save the Fusion results')
            
    
    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt