python tools/lazyconfig_train_net_nia_segmentation.py \
    --config-file projects/ViTDet/configs/NIA/mask_rcnn_vitdet_b_100ep.py \
    --eval-only \
    "dataloader.test.dataset.names='nia_test'" \
    "train.init_checkpoint='output/finetuned/model_final.pth'"
