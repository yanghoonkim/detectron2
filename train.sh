python tools/lazyconfig_train_net_nia_segmentation.py \
    --config-file projects/ViTDet/configs/NIA/mask_rcnn_vitdet_b_100ep.py \
    "train.output_dir='/root/detectron2/output'" \
    "train.init_checkpoint='/root/detectron2/nia/model_final_61ccd1.pkl'"