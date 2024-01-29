from multiprocessing import Pool
from pathlib import Path
import pickle
from detectron2.structures import BoxMode
import pandas as pd
import json
import numpy as np
import math

from nia.nia_dataset_reader import (
    NiaDataPathExtractor,
    DataFrameSplitter,
    NiaDataPathProvider,
)

from nia.poly2bitmask import build_segmentation_from_multipoly

BASE_PATH = Path('/home/detectron2/datasets/nia/')
ANNO_PATH = BASE_PATH / '2.라벨링데이터'
IMG_PATH = BASE_PATH / '1.원천데이터'
TRAIN_PAIRS_LIST = BASE_PATH / 'visible_train_pairs.pkl'
VALID_LABEL_PATH = BASE_PATH / 'visible_valid_label.json'
TEST_LABEL_PATH = BASE_PATH / 'visible_test_label.json'
BUG_LIST = Path('/home/detectron2/nia/img_w_bug.pkl')


# categories 정의
categories = [{'id': 0,
   'name': 'Car',
   'category_id': '36e54347-aa45-4a9b-8c85-f29b10fd962e',
   'supercategory': 'Vehicle'},
  {'id': 1,
   'name': 'Two-wheel Vehicle',
   'category_id': 'e90a7f26-4925-48ae-b07c-421d766b8eaa',
   'supercategory': 'Vehicle'},
  {'id': 2,
   'name': 'Personal Mobility',
   'category_id': '6cb349a7-9c5f-4374-a5cd-c39db76ea58b',
   'supercategory': 'Vehicle'},
  {'id': 3,
   'name': 'TruckBus',
   'category_id': 'b1fb8d37-6774-4304-a22a-38f468187b7c',
   'supercategory': 'Vehicle'},
  {'id': 4,
   'name': 'Kid student',
   'category_id': '5e6cf847-77d7-4214-a59e-246645106eda',
   'supercategory': 'Pedestrian'},
  {'id': 5,
   'name': 'Adult',
   'category_id': 'd6be826f-eac8-45a9-82c7-17622c34a4de',
   'supercategory': 'Pedestrian'},
  {'id': 6,
   'name': 'Traffic Sign',
   'category_id': '0e935d3a-6e5d-4fd0-85aa-ef1245b91caf',
   'supercategory': 'Outdoor'},
  {'id': 7,
   'name': 'Traffic Light',
   'category_id': 'fc75e996-8759-4fa0-a7be-4e239c31e893',
   'supercategory': 'Outdoor'},
  {'id': 8,
   'name': 'Speed bump',
   'category_id': '3c88f18a-a268-42bf-9a4b-76fe2b356e9d',
   'supercategory': 'Outdoor'},
  {'id': 9,
   'name': 'Parking space',
   'category_id': 'c1a0185c-c4d0-4087-9448-92b10d3a387a',
   'supercategory': 'Outdoor'},
  {'id': 10,
   'name': 'Crosswalk',
   'category_id': '409123c5-26ae-4509-9485-52f8391d7dfd',
   'supercategory': 'Outdoor'}]

id_to_contiguous_id = {3:0, 2:1, 99:2, 8:3, 97:4, 98:5, 12:6, 10:7, 52:8, 51:9, 100:10}
original_ids = list(id_to_contiguous_id.keys())

# 가시광 데이터 필터링
def is_visible_data(item):
    cond = item.match('image_B/*.png*') or item.match('image_F/*.png*') or item.match('image_L/*.png*') or item.match('image_R/*.png*')
    return cond

def to_frame(pairs):
    df = pd.DataFrame(pairs, columns=['imgpath', 'annopath'])
    df.index = df.imgpath.apply(lambda x: x.split('/')[-1])
    df.index.name = 'filename'
    return df


def process_anno4nia(annotations):
    processed_annos = list()
    for anno in annotations:
        anno_dict = dict()
        anno_dict['image_id'] = anno['image_id']
        anno_dict['iscrowd'] = anno['iscrowd']
        anno_dict['bbox'] = anno['bbox']
        anno_dict['category_id'] = id_to_contiguous_id[anno['category_id']]
        anno_dict['segmentation'] = build_segmentation_from_multipoly(anno['segmentation'])
        anno_dict['bbox_mode'] = BoxMode.XYWH_ABS
        anno_dict['area'] = anno['area']
        processed_annos.append(anno_dict)
    
    return processed_annos


# visible 데이터 오류 검출
# 열영상 데이터 annotations에서 category_id가 [3, 2, 99, 8, 97, 98, 12, 10, 52, 51, 100] 범주를 초과한 경우들 제외하기
# json 파일에서 필요한 정보만 추출: 'images', 'annotations'
def make_dict(df):
    anno_images = list()
    anno_annotations = list()

    for filename, item in zip(df.imgpath, df.annopath):
        issue_flag = False
        with open(item) as f:
            item_json = json.load(f)
        for anno in item_json['annotations']:
            if anno['category_id'] not in original_ids:
                issue_flag = True
                break
        if not issue_flag:
            anno_images.extend(item_json['images'])
            anno_images[-1]['file_name'] = Path(filename).relative_to(IMG_PATH.as_posix()).as_posix()
            processed_annos = process_anno4nia(item_json['annotations'])
            anno_annotations.extend(processed_annos)
    
    dict_ = {'categories': categories, 'images': anno_images, 'annotations': anno_annotations}

    return dict_


def process_single_file(annopath):
    issue_flag = False
    with open(annopath) as f:
        item_json = json.load(f)
    for anno in item_json['annotations']:
        if anno['category_id'] not in original_ids:
            issue_flag = True
            break    
    if not issue_flag:
        anno_img = item_json['images'][0]
        anno_img['file_name'] = Path(annopath).relative_to(ANNO_PATH.as_posix()).as_posix().rstrip('.json')
        processed_annos = process_anno4nia(item_json['annotations'])
        
        return [[anno_img], processed_annos]
    else:
        return [[], []]
        
    
def make_dict_mp(df):
    mp_workers = Pool()
    anno_images = list()
    anno_annotations = list()
                
    packed_item = mp_workers.map(process_single_file, df.annopath)
    anno_images = [item[0] for item in packed_item]
    anno_images = np.concatenate(anno_images).tolist()
    anno_annotations = [item[1] for item in packed_item]
    anno_annotations = np.concatenate(anno_annotations).tolist()

    dict_ = {'categories': categories, 'images': anno_images, 'annotations': anno_annotations}

    return dict_


def get_essential_data():
    if not TRAIN_PAIRS_LIST.exists() or not VALID_LABEL_PATH.exists() or not TEST_LABEL_PATH.exists():
        print('Get essential data...(it may cost several minutes)')

        with BUG_LIST.open('rb') as f:
            img_w_bug = pickle.load(f)

        path_provider = NiaDataPathProvider(
            reader=NiaDataPathExtractor(dataset_dir=BASE_PATH.as_posix()),
            exclude_filenames=img_w_bug,
        )
        train_path_pairs = path_provider.get_split_data_list(channels=["image_B", "image_F", "image_L", "image_R"], splits="train")
        valid_path_pairs = path_provider.get_split_data_list(channels=["image_B", "image_F", "image_L", "image_R"], splits="valid")
        test_path_pairs = path_provider.get_split_data_list(channels=["image_B", "image_F", "image_L", "image_R"], splits="test")

        df_visible_valid = to_frame(valid_path_pairs)
        valid_dict = make_dict_mp(df_visible_valid)

        df_visible_test = to_frame(test_path_pairs)
        test_dict = make_dict_mp(df_visible_test)

        # annotation id 중복 이슈 해결
        anno_id = 0
        for idx, item in enumerate(valid_dict['annotations']):
            valid_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1

        for idx, item in enumerate(test_dict['annotations']):
            test_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1       
        
        # 저장
        with open(TRAIN_PAIRS_LIST, 'wb') as f:
            pickle.dump(train_path_pairs, f)

        with VALID_LABEL_PATH.open('w') as f:
            json.dump(valid_dict, f)
        
        with TEST_LABEL_PATH.open('w') as f:
            json.dump(test_dict, f)
        

    else:
        print('Load essential data...')
