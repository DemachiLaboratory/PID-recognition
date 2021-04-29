import os
import cv2
import math
import shutil
import numpy as np
import glob
from tqdm import tqdm

class SlidingWindow:
    def __init__(self, root_dir=None, save_dir=None, temp_dir=None, resize=(None, None), inter_ratio=None, infer_mode=False):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.temp_dir = temp_dir
        self.resize = resize
        self.inter_ratio = inter_ratio
        self.infer_mode = infer_mode

    def split(self):
        resize_h, resize_w = self.resize
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        for img_path in glob.glob(os.path.join(self.root_dir, '*.jpg')):
            print(img_path)
            img = cv2.imread(img_path)
            ori_h, ori_w, _ = img.shape
            row_count = math.ceil(((ori_h - resize_h) / ((1 - self.inter_ratio) * resize_h)) + 1 )
            col_count = math.ceil(((ori_w - resize_w) / ((1 - self.inter_ratio) * resize_w)) + 1 )

            # Extend the image if necessary
            img = cv2.copyMakeBorder(src=img,
                                     top=0, 
                                     bottom=int(resize_h * row_count - ori_h - self.inter_ratio * resize_h * (row_count - 1)),
                                     left=0,
                                     right=int(resize_w * col_count - ori_w - self.inter_ratio * resize_w * (col_count - 1)),
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=0)
            
            for i in range(row_count):
                for j in range(col_count):
                    print('({0}, {1})'.format(i, j))
                    ori_spl_border = np.zeros(5) # The border of the split sample ([1:])
                    ori_spl_border[1] = (1 - self.inter_ratio) * resize_w * j
                    ori_spl_border[2] = ori_spl_border[1] + resize_w
                    ori_spl_border[3] = (1 - self.inter_ratio) * resize_h * i
                    ori_spl_border[4] = ori_spl_border[3] + resize_h
                    spl_border = ori_spl_border.copy()

                    if not self.infer_mode:
                        with open(os.path.splitext(img_path)[0] + '.txt', 'r') as f:
                            lines = f.readlines()
                            # Extend the split sample according to the annotations
                            for line in lines:
                                flag = False
                                rel_xyxy = list(map(float, line.strip().split(' ')))    # Relative coordinates
                                abs_xyxy = [rel_xyxy[0], 
                                            (rel_xyxy[1] - rel_xyxy[3] / 2) * ori_w,
                                            (rel_xyxy[1] + rel_xyxy[3] / 2) * ori_w,
                                            (rel_xyxy[2] - rel_xyxy[4] / 2) * ori_h,
                                            (rel_xyxy[2] + rel_xyxy[4] / 2) * ori_h]    # Absoluate coordinates
                                
                                if (((ori_spl_border[2] > abs_xyxy[1] > ori_spl_border[1]) or 
                                     (ori_spl_border[1] < abs_xyxy[2] < ori_spl_border[2])) and 
                                    ((ori_spl_border[4] > abs_xyxy[3] > ori_spl_border[3]) or 
                                     (ori_spl_border[3] < abs_xyxy[4] < ori_spl_border[4]))): 
                                    flag = True 
                                elif ((abs_xyxy[1] > ori_spl_border[1]) and (abs_xyxy[2] < ori_spl_border[2]) and
                                      (abs_xyxy[3] < ori_spl_border[3]) and (abs_xyxy[4] > ori_spl_border[4])):
                                    flag = True
                                elif ((abs_xyxy[3] > ori_spl_border[3]) and (abs_xyxy[4] < ori_spl_border[4]) and
                                      (abs_xyxy[1] < ori_spl_border[1]) and (abs_xyxy[2] > ori_spl_border[2])):
                                    flag = True

                                # Extend the split sample border by comparing to the absolute coordinates of annotations
                                if flag: 
                                    for idx in range(1, 5):
                                        spl_border[idx] = max(spl_border[idx], abs_xyxy[idx]) if idx % 2 == 0 else min(spl_border[idx], abs_xyxy[idx])

                            adp_w = spl_border[2] - spl_border[1] # Adapted width of the split sample
                            adp_h = spl_border[4] - spl_border[3] # Adapted height of the split sample

                            # Save the adapted border of the split sample
                            adp_border_dir = os.path.join(self.save_dir, 'adapted_borders')
                            if not os.path.isdir(adp_border_dir):
                                os.makedirs(adp_border_dir)
                            adp_border_path = os.path.join(adp_border_dir, '{0}_{1}_{2}.txt'.format(os.path.splitext(os.path.split(img_path)[1])[0], i, j))
                            with open(adp_border_path, 'a+') as cf:
                                cf.write(' '.join(list(map(str, spl_border[1:]))))
                                cf.write('\n')
                                
                            # Save the annotations of each split sample
                            for line in lines:
                                flag = False
                                rel_xyxy = list(map(float, line.strip().split(' ')))    # Relative coordinates
                                abs_xyxy = [rel_xyxy[0], 
                                            (rel_xyxy[1] - rel_xyxy[3] / 2) * ori_w,
                                            (rel_xyxy[1] + rel_xyxy[3] / 2) * ori_w,
                                            (rel_xyxy[2] - rel_xyxy[4] / 2) * ori_h,
                                            (rel_xyxy[2] + rel_xyxy[4] / 2) * ori_h]    # Absoluate coordinates

                                if (abs_xyxy[1] > spl_border[1] and abs_xyxy[2] < spl_border[2] and 
                                    abs_xyxy[3] > spl_border[3] and abs_xyxy[4] < spl_border[4]):
                                    flag = True
                                elif (((spl_border[2] > abs_xyxy[1] > spl_border[1]) or 
                                       (spl_border[1] < abs_xyxy[2] < spl_border[2])) and 
                                      ((spl_border[4] > abs_xyxy[3] > spl_border[3]) or 
                                       (spl_border[3] < abs_xyxy[4] < spl_border[4]))):
                                    flag = True
                                    for idx in range(1, 5):
                                        abs_xyxy[idx] = min(spl_border[idx], abs_xyxy[idx]) if idx % 2 == 0 else max(spl_border[idx], abs_xyxy[idx])

                                if flag:
                                    resize_ann = [int(rel_xyxy[0]),
                                                  ((abs_xyxy[1] + abs_xyxy[2]) / 2 - spl_border[1]) / adp_w,
                                                  ((abs_xyxy[3] + abs_xyxy[4]) / 2 - spl_border[3]) / adp_h,
                                                  (abs_xyxy[2] - abs_xyxy[1]) / adp_w,
                                                  (abs_xyxy[4] - abs_xyxy[3]) / adp_h]
                                    save_path = os.path.join(self.save_dir, '{0}_{1}_{2}.txt'.format(os.path.splitext(os.path.split(img_path)[1])[0], i, j))
                                    with open(save_path, 'a+') as af:
                                        af.write(' '.join(list(map(str, resize_ann))))
                                        af.write('\n')
                                    
                    # Save the split sample
                    spl_img = img[int(spl_border[3]):int(spl_border[4]), int(spl_border[1]):int(spl_border[2])]
                    save_path = os.path.join(self.save_dir, '{0}_{1}_{2}.jpg'.format(os.path.splitext(os.path.split(img_path)[1])[0], i, j))
                    cv2.imwrite(save_path, spl_img)

    def merge(self, filename):
        global_border = self.__get_global_border(filename)
        global_w = global_border[1] - global_border[0]
        global_h = global_border[3] - global_border[2]
        global_img = np.zeros((int(global_border[3]), int(global_border[1]), 3), dtype=np.uint8)
        for img_path in sorted(glob.glob(os.path.join(self.save_dir, '{}_?_?.jpg'.format(os.path.splitext(filename)[0])))):
            i, j = self.__get_row_col_idx(img_path)
            spl_img = cv2.imread(img_path)
            adp_border_path = os.path.join(self.save_dir, 'adapted_borders', '{}.txt'.format(os.path.splitext(os.path.split(img_path)[1])[0]))
            with open(adp_border_path, 'r') as f:
                lines = f.readlines()
                spl_border = list(map(float, lines[0].strip().split(' ')))
                global_img[int(spl_border[2]):int(spl_border[3]), int(spl_border[0]):int(spl_border[1])] = spl_img
        
        save_dir = os.path.join(self.temp_dir, 'merge')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, global_img)

    def show(self):
        save_dir = os.path.join(self.temp_dir, 'split samples')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for img_path in tqdm(glob.glob(os.path.join(self.save_dir, '*.jpg'))):
            img = cv2.imread(img_path)
            h, w, _ = img.shape

            with open(os.path.splitext(img_path)[0] + '.txt', 'r') as f:
                for line in f.readlines():
                    rel_xyxy = list(map(float, line.strip().split(' ')))    # Relative coordinates
                    abs_xyxy = [rel_xyxy[0], 
                                int((rel_xyxy[1] - rel_xyxy[3] / 2) * w),
                                int((rel_xyxy[1] + rel_xyxy[3] / 2) * w),
                                int((rel_xyxy[2] - rel_xyxy[4] / 2) * h),
                                int((rel_xyxy[2] + rel_xyxy[4] / 2) * h)]    # Absoluate coordinates
                                
                    show = cv2.rectangle(img, (abs_xyxy[1], abs_xyxy[3]), (abs_xyxy[2], abs_xyxy[4]), (0,255,0), 2)
                    save_path = os.path.join(save_dir, 'output_{0}'.format(os.path.split(img_path)[1]))
                    cv2.imwrite(save_path, show)
    
    def __get_global_border(self, filename):
        global_border = np.zeros(4)
        for adp_border_path in glob.glob(os.path.join(self.save_dir, 'adapted_borders', '{}_?_?.txt'.format(os.path.splitext(filename)[0]))):
            with open(adp_border_path, 'r') as f:
                lines = f.readlines()
                spl_border = list(map(float, lines[0].strip().split(' ')))
                for idx in range(4):
                    global_border[idx] = min(spl_border[idx], global_border[idx]) if idx % 2 == 0 else max(spl_border[idx], global_border[idx])

        return global_border

    def __get_row_col_idx(self, img_path):
        idxs = os.path.splitext(os.path.split(img_path)[1])[0].strip().split('_')
        return idxs[1], idxs[2]
        
if __name__ == '__main__':
    sw = SlidingWindow(root_dir='/mnt/database/Dataset/PID_yolo/train', 
                       save_dir='/mnt/database/Experiments/20210426_yolo/save/',
                       temp_dir='/mnt/database/Experiments/20210426_yolo/temp/',
                       resize=(1280, 1280), 
                       inter_ratio=0.15)
    #sw.split()
    #sw.show()
    sw.merge('H21-420-01-Z04-01.jpg')