#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from pathlib import Path
import pandas as pd
import math
import numpy as np
import json
from detectron2.data.datasets import register_coco_instances

from nia.utils import *

logger = logging.getLogger("detectron2")

BASE_PATH = Path('/home/detectron2/datasets/nia/')
ANNO_PATH = BASE_PATH / '라벨링데이터'
COLL_PATH = BASE_PATH / '원천데이터'
TRAIN_LABEL_PATH = BASE_PATH / 'visible_train_label.json'
VALID_LABEL_PATH = BASE_PATH / 'visible_valid_label.json'
TEST_LABEL_PATH = BASE_PATH / 'visible_test_label.json'



def split_data():

    if (not TRAIN_LABEL_PATH.exists()) or (not VALID_LABEL_PATH.exists()) or (not TEST_LABEL_PATH.exists()):
        print('[DATA SPLIT] Splitting data...')
        img_paths = list(COLL_PATH.rglob('*.png'))
        anno_paths = list(ANNO_PATH.rglob('*.json'))

        # 이미지와 annotation이 동시에 존재하는 파일만 필터링
        visible_anno_paths = list(filter(is_visible_data, anno_paths))
        visible_anno_paths = list(filter(lambda x: '._' not in x.as_posix(), visible_anno_paths))
        visible_anno_names = [item.name for item in visible_anno_paths]
        visible_anno_names_wo_json = [item.rstrip('.json') for item in visible_anno_names]

        visible_img_paths = list(filter(is_visible_data, img_paths))
        visible_img_names = [item.name for item in visible_img_paths]

        df_visible_img = pd.DataFrame({'filename': visible_img_names, 'imgpath': visible_img_paths}).set_index('filename')
        df_visible_anno = pd.DataFrame({'filename': visible_anno_names_wo_json, 'annopath': visible_anno_paths}).set_index('filename')

        df_visible = pd.concat([df_visible_img, df_visible_anno], axis=1).dropna(how='any')
        df_visible = df_visible.sample(frac=1, random_state=0) # random shuffle

        # visible 데이터 오류 검출
        # 가시광 데이터 annotations에서 segmentation이 None으로 표기된 경우들을 제외하기
        # json 파일에서 필요한 정보만 추출: 'images', 'annotations'
        anno_images = list()
        anno_annotations = list()

        for item in df_visible.annopath:
            none_flag = False
            item_json = json.load(item.open())
            for anno in item_json['annotations']:
                if anno['segmentation'] is None:
                    none_flag = True
                    break
            if not none_flag:
                anno_images.append(item_json['images'])
                anno_annotations.append(item_json['annotations'])

        # data split
        ratio = [8, 1, 1] # train / valid / test
        ratio = [item / sum(ratio) for item in ratio]    

        total_len = len(anno_images)
        train_len = math.floor(total_len * ratio[0])
        valid_len = train_len + math.floor(total_len * ratio[1])

        train_dict = dict()
        valid_dict = dict()
        test_dict = dict()

        train_dict['categories'] = categories
        train_dict['images'] = anno_images[:train_len]
        train_dict['images'] = np.concatenate(train_dict['images']).tolist() # [[1],[2],[3]] -> [1,2,3]
        train_dict['annotations'] = anno_annotations[:train_len] 
        train_dict['annotations'] = np.concatenate(train_dict['annotations']).tolist() # [[1,2],[3,4],[5,6]] -> [1,2,3,4,5,6]

        valid_dict['categories'] = categories
        valid_dict['images'] = anno_images[train_len:valid_len]
        valid_dict['images'] = np.concatenate(valid_dict['images']).tolist()
        valid_dict['annotations'] = anno_annotations[train_len:valid_len]
        valid_dict['annotations'] = np.concatenate(valid_dict['annotations']).tolist()

        test_dict['categories'] = categories
        test_dict['images'] = anno_images[valid_len:]
        test_dict['images'] = np.concatenate(test_dict['images']).tolist()
        test_dict['annotations'] = anno_annotations[valid_len:]
        test_dict['annotations'] = np.concatenate(test_dict['annotations']).tolist()


        # annotation id 중복 이슈 해결
        #  + category_id가 잘못 된 경우 수정 (0 -> 1)
        anno_id = 0
        for idx, item in enumerate(train_dict['annotations']):
            train_dict['annotations'][idx]['id'] = anno_id
            train_dict['annotations'][idx]['category_id'] = 1 if item['category_id'] == 0 else item['category_id']
            anno_id += 1
        for idx, item in enumerate(valid_dict['annotations']):
            valid_dict['annotations'][idx]['id'] = anno_id
            valid_dict['annotations'][idx]['category_id'] = 1 if item['category_id'] == 0 else item['category_id']
            anno_id += 1
        for idx, item in enumerate(test_dict['annotations']):
            test_dict['annotations'][idx]['id'] = anno_id
            test_dict['annotations'][idx]['category_id'] = 1 if item['category_id'] == 0 else item['category_id']
            anno_id += 1



        # folder hierarchy가 다를수도 있기 때문에 실제 file_name으로 바꿔주기
        for idx, item in enumerate(train_dict['images']):
            file_name_wo_dir = item['file_name'].split('/')[-1]
            real_file_name = df_visible.loc[file_name_wo_dir, 'imgpath'].relative_to('/home/detectron2/datasets/nia/collections/').as_posix()
            train_dict['images'][idx]['file_name'] = real_file_name

        for idx, item in enumerate(valid_dict['images']):
            file_name_wo_dir = item['file_name'].split('/')[-1]
            real_file_name = df_visible.loc[file_name_wo_dir, 'imgpath'].relative_to('/home/detectron2/datasets/nia/collections/').as_posix()
            valid_dict['images'][idx]['file_name'] = real_file_name
            

        for idx, item in enumerate(test_dict['images']):
            file_name_wo_dir = item['file_name'].split('/')[-1]
            real_file_name = df_visible.loc[file_name_wo_dir, 'imgpath'].relative_to('/home/detectron2/datasets/nia/collections/').as_posix()
            test_dict['images'][idx]['file_name'] = real_file_name
        
        with TRAIN_LABEL_PATH.open('w') as f:
            json.dump(train_dict, f)
        
        with VALID_LABEL_PATH.open('w') as f:
            json.dump(valid_dict, f)

        with TEST_LABEL_PATH.open('w') as f:
            json.dump(test_dict, f) 
    else:
        print('[DATA SPLIT] Load existing files...')


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    
    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)

    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    get_valid_data()
    register_coco_instances('nia_val', {}, VALID_LABEL_PATH, COLL_PATH)
    cfg = LazyConfig.load(args.config_file)
    cfg.dataloader.train.mapper['instance_mask_format'] = 'bitmask'
    cfg.dataloader.test.mapper['instance_mask_format'] = 'bitmask'
    cfg.dataloader.test.dataset.names = 'nia_val'

    cfg.train.init_checkpoint = '/home/detectron2/nia/model_final_61ccd1.pkl'
    cfg.train.eval_period = 20
    cfg.dataloader.train.total_batch_size = 1
    cfg.model.roi_heads.num_classes = 11
    cfg.optimizer.lr = 1e-5
    cfg.train.output_dir = './output/segmentation'

    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
