from mmdet.apis import init_detector, inference_detector
import mmcv
# import os


# # モデルの設定ファイルと学習済モデルへのパスを指定する
# config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = './checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
# Choose to use a config and initialize the detector
config_file = './configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# Setup a checkpoint file to load
checkpoint_file = './checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# モデルの初期化
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# path = os.getcwd()

# print(path)
# テストする画像を指定
img = './demo/demo.jpg'

# テスト
result = inference_detector(model, img)

# 結果の保存
model.show_result(img, result, out_file='result_faster_rcnn.jpg')