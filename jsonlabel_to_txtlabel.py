import cv2
import os
import json

img_dir = r'C:\Users\beyon\Desktop\carplate_regression_20191231'
# json_dir =r'C:\Users\beyon\Desktop\carplate_polygon_label_20191229\annotation'
img_files = os.listdir(img_dir)
# txt_dir = r'C:\Users\beyon\Desktop\carplate_polygon_label_20191229\txt'

for img_file in img_files:
    name, ext = os.path.splitext(img_file)
    if ext != '.jpg':
        continue
    img = cv2.imread(os.path.join(img_dir, img_file))
    h, w, _ = img.shape

    try:
        with open(os.path.join(img_dir, name+'.json'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        objects = json_data['shapes']
        obj = objects[0]
        pts = obj['points']
        str_list = ['4']
        xs = [str(pt[0]/w) for pt in pts]
        ys = [str(pt[1]/h) for pt in pts]
        str_list = str_list + xs + ys
        write_str = ','.join(str_list)
        write_str = write_str+',,'
        with open(os.path.join(img_dir, name+'.txt'), 'w', encoding='utf-8') as f:
            f.write(write_str)
    except:
        print(img_file)
