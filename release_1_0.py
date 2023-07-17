import os
import random

from PIL import Image
import Augmentor

import numpy as np
from tqdm import tqdm
import time


def extract(file_path, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('running program ...')
    lt = os.listdir(file_path)
    ##根据名称进行提取
    label_name = 'label.png'
    img_name = 'img.png'
    count = 0
    for file_name in lt:
        count += 1

        label_path = os.path.join(file_path, file_name, label_name)
        img_path = os.path.join(file_path, file_name, img_name)

        label_file = Image.open(label_path)
        save_label_path = save_path + '/' + 'label' + '/'

        if not os.path.exists(save_label_path):
            os.makedirs(save_label_path)

        label_file.save(save_label_path + f'{count}.png')


        img_file = Image.open(img_path)
        save_img_path = save_path + '/' + 'source' + '/'

        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        img_file.save(save_img_path + f'{count}.png')
    print('program extract was finished successfully')
        ##输出的文件地址
        ##label地址： save_label_path
        ##img地址： save_img_path


def transfer(file_path, save_file_path):
    print('running program transfer ...')
    file_list = os.listdir(file_path)

    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    for i in file_list:
        name = i.split('.')
        # i 就是文件的名称


        save_path = os.path.join(save_file_path, name[0] + '.jpg')


        f = Image.open(os.path.join(file_path, i))

        f = f.convert('RGB')

        f.save(save_path)

    print("the program was finished successfully")


class Data_Enhance:
    def __init__(self,img_path,ground_truth):
        self.img_path = img_path

        self.ground_truth = ground_truth
        self.p = Augmentor.Pipeline(self.img_path)

        self.p.ground_truth(self.ground_truth)

    def up_down_reserve(self):
        self.p.flip_top_bottom(probability=1)

        self.p.process()

    def left_right_reserve(self):
        self.p.flip_left_right(probability=1)

        self.p.process()

    def point_reserve(self):
        self.p.flip_left_right(probability=1)
        self.p.flip_top_bottom(probability=1)

        self.p.process()

    def random_color(self):
        self.p.flip_left_right(probability=1)
        self.p.flip_top_bottom(probability=1)

        self.p.random_color(probability=1, min_factor=0, max_factor=1)

        self.p.process()

    def random_brightness(self):
        self.p.flip_left_right(probability=1)
        self.p.flip_top_bottom(probability=1)

        self.p.random_brightness(probability=1, min_factor=0.7, max_factor=1.2)

        self.p.process()





##将数据增强后的文件进行重命名
##output_path：含有原图和掩码图数据增强后的图片
##source_path：通过路径得出原图的数量
##取得原图数量后按顺序将后面增强的图片加上
##save_mask_path：保存重命名后的掩码图
def out_put_rename(output_path, source_path ,save_mask_path ):
    file_name = os.listdir(output_path)
    file_num = len(file_name)

    name_split = output_path.split('/')

    source_num = len(os.listdir(source_path))
    mask_count = 0
    source_count = 0
    for i in range(file_num):

        name = file_name[i]

        if name_split[-2] + '_original' in name:
            src_path = os.path.join(output_path, name)

            source_count += 1
            os.rename(src_path, source_path +'/' + str(source_num + 1 + source_count) + '.png')

        if '_groundtruth_(1)_' + name_split[-2] in name:
            mask_count += 1

            src_path = os.path.join(output_path, name)

            os.rename(src_path, save_mask_path + '/' + str(source_num + 1 + mask_count) + '.png')

##将刚刚分好类的24位mask改为8位深

#mask_path:需要进行转换的掩码图的路径
#standar_path:从json文件夹中提取的标准掩码图
def to_8_bit(mask_path,standar_path):
    file_name = os.listdir(mask_path)

    standar_img = Image.open(standar_path + '/' + 'label.png')

    ##获取调色板模版
    img_palette = standar_img.getpalette()

    for name in file_name:
        img = Image.open(os.path.join(mask_path, name))

        if img.mode == 'P' and len(img.split()) == 1:
            #如果已经是调色板模式，直接跳过
            continue
        else:
            img_8bits = img.convert('P')
            ##将标准的调色板模版导入
            img_8bits.putpalette(img_palette)
            ##进度条
            pbar = tqdm(total=img_8bits.size[0] == img_8bits.size[1])
            ndimg = np.array(img_8bits)
            x, y = (ndimg > 0).nonzero()
            for i in range(len(x)):
                pbar.update(1)
                img_8bits.putpixel((y[i], x[i]), 1)
            os.remove(mask_path + '/' + name)
            img_8bits.save(mask_path + '/' + name)

    print('finish!')

#parent_path:掩码图路径的父目录
def voc_annotation(parent_path):
    trainval_percent = 1
    train_percent = 0.9

    random.seed(0)
    print("Generate txt in ImageSets.")

    mask_path = os.path.join(parent_path, 'label')

    txt_path = os.path.join(parent_path, 'txt')

    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    mask_name = os.listdir(mask_path)
    volid_name = []
    for i in mask_name:
        if '.png' in i :
            volid_name.append(i)

    num = len(volid_name)

    #
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(txt_path, 'trainval.txt'), 'w')
    ftest = open(os.path.join(txt_path, 'test.txt'), 'w')
    ftrain = open(os.path.join(txt_path, 'train.txt'), 'w')
    fval = open(os.path.join(txt_path, 'val.txt'), 'w')

    for i in list:
        name = volid_name[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")

    classes_nums = np.zeros([256], int)
    for i in tqdm(list):
        name = volid_name[i]
        png_file_name = os.path.join(mask_path, name)
        # png_file_name = os.path.join(save_path, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。" % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))
            # print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print('voc_annotation program was finished')


def main(json_path ,save_path):
    time_start = time.time()
    ##json文件夹路径
    ##保存路径
    extract(json_path, save_path)

    file = save_path + '/' + 'label'
    save = save_path + '/' + '24_bit'
    file2 = save_path + '/' + 'source'
    save2 = save_path + '/' + 'source_jpg'
    transfer(file, save)
    transfer(file2, save2)

    example = Data_Enhance(save2, save)
    example.up_down_reserve()
    example.left_right_reserve()
    example.random_color()
    example.point_reserve()
    example.random_brightness()

    output_path = save2 + '/' + 'output'
    out_put_rename(output_path, file2, file)

    standar_path = json_path + '/' + '1_json'

    to_8_bit(file, standar_path)

    voc_annotation(save_path)

    time_end = time.time()
    time_sum = time_end - time_start
    print(f'所花费的时间：{time_sum}秒')


main(r'F:\front_crack_json_7_17', 'F:/save_json_test_7_17')


'''
##json文件夹路径
json_path = 'F:/back_scrath/Back_Scratch/json'
##保存路径
save_path = 'F:/back_scrath/Back_Scratch/test'
extract(json_path, save_path)
##保存原图，掩码图
file = 'F:/back_scrath/Back_Scratch/test/label'
file2 = 'F:/back_scrath/Back_Scratch/test/source'
save = 'F:/back_scrath/Back_Scratch/test/24_bit'
save2 = 'F:/back_scrath/Back_Scratch/test/source_jpg'
transfer(file, save)
transfer(file2, save2)


##原图保存的地址，需要数据增强的掩码图的地址
img_path = 'F:/back_scrath/Back_Scratch/test/source_jpg'
ground_truth = 'F:/back_scrath/Back_Scratch/test/24_bit'

##数据增强
example = Data_Enhance(img_path, ground_truth)
example.up_down_reserve()
example.left_right_reserve()
example.random_color()
example.point_reserve()
example.random_brightness()

##output_path:产生的数据增强图片的地址，包含增强后的原图和掩码图
output_path = 'F:/back_scrath/Back_Scratch/test/source_jpg/output'
##原图保存路径
source_path = "F:/back_scrath/Back_Scratch/test/source"
##掩码图保存路径
save_mask_path = 'F:/back_scrath/Back_Scratch/test/label'

out_put_rename(output_path, source_path, save_mask_path)


mask_path = 'F:/back_scrath/Back_Scratch/test/label'
standar_path = 'F:/back_scrath/Back_Scratch/json/5_json'

to_8_bit('F:/back_scrath/Back_Scratch/test/label'
         ,'F:/back_scrath/Back_Scratch/json/5_json')

parent_path = 'F:/back_scrath/Back_Scratch/test'

voc_annotation('F:/back_scrath/Back_Scratch/test')

time_end = time.time()

time_sum = time_end - time_start

print(f'所花费的时间：{time_sum}s') 
'''

