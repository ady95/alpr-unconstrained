import sys
import os
import cv2
import traceback

from src.keras_utils import load_model, detect_lp
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.label import Shape, writeShapes


def adjust_pts(pts, lroi):
    return pts*lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


if __name__ == '__main__':

    try:
        # input_dir  = sys.argv[1]
        # input_dir  = r'C:\Users\beyon\Desktop\license-plate-test'
        # input_dir = r'/home/nextlab-ai/license-plate/images/test/license-plate-test-org'
        input_dir = r'D:\GIT\alpr-unconstrained_ady95\images\test\license-plate-test-org'
        # output_dir = r'C:\Users\beyon\Desktop\LP_result'
        # if not os.path.exists(output_dir):
        # 	os.makedirs(output_dir)

        lp_threshold = .5

        # wpod_net_path = sys.argv[2]
        # wpod_net_path = r'C:\Users\beyon\Desktop\license-plate\models\car-plate-model\car-plate-model_final'
        script_path = os.path.dirname(os.path.realpath(__file__))
        wpod_net_path = os.path.join(
            script_path, 'models/car-plate-model/car-plate-model_final')
        wpod_net = load_model(wpod_net_path)

        imgs_paths = glob('%s/*.jpg' % input_dir)

        print('Searching for license plates using WPOD-NET')
        # print(imgs_paths)

        for i, img_path in enumerate(imgs_paths):
            print('\t Processing %s' % img_path)
            bname = splitext(basename(img_path))[0]
            Ivehicle = cv2.imread(img_path)

            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            side = int(ratio*288.)
            bound_dim = min(side + (side % (2**4)), 608)
            print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

            Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(
                Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold)

            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                img = (Ilp * 255.).astype('uint8')
                cv2.imshow('origin', Ivehicle)
                cv2.imshow('img', img)
                cv2.waitKey(0)

                # s = Shape(Llp[0].pts)

                # cv2.imwrite('%s/%s.jpg' % (output_dir,bname),Ilp*255.)
                # writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
