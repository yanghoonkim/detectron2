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
COLL_PATH = BASE_PATH / '1.원천데이터'
TRAIN_LABEL_PATH = BASE_PATH / 'visible_train_label.json'
VALID_LABEL_PATH = BASE_PATH / 'visible_valid_label.json'
TEST_LABEL_PATH = BASE_PATH / 'visible_test_label.json'


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
            anno_images[-1]['file_name'] = Path(filename).relative_to('/home/detectron2/datasets/nia/1.원천데이터/').as_posix()
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
        anno_img['file_name'] = Path(annopath).relative_to('/home/detectron2/datasets/nia/2.라벨링데이터/').as_posix().rstrip('.json')
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


def get_evaluation_data(split_name='valid'):
    if split_name == 'valid':
        EVAL_LABEL_PATH = VALID_LABEL_PATH
    else:
        EVAL_LABEL_PATH = TEST_LABEL_PATH

    if not EVAL_LABEL_PATH.exists():
        print('Get Json file for evaluation...')

        with open('/home/detectron2/nia/img_w_bug.pkl', 'rb') as f:
            img_w_bug = pickle.load(f)

        path_provider = NiaDataPathProvider(
            reader=NiaDataPathExtractor(dataset_dir="/home/detectron2/datasets/nia/",
                                        exclude_filenames=img_w_bug),
            splitter=DataFrameSplitter(
                groups=["channel", "collector", "sensor", "code_1", "code_2", "timeslot", "weather"],
                splits=["train", "valid", "test"],
                ratios=[8, 1, 1],
                #splits=['train', 'valid'],
                #ratios=[79,1],
                seed=231111,
            ),
            channels=["image_B", "image_F", "image_L", "image_R"],
        )

        path_pairs = path_provider.get_split_data_list(split_name)
        df_visible_eval = to_frame(path_pairs)
        eval_dict = make_dict_mp(df_visible_eval)

        # annotation id 중복 이슈 해결
        anno_id = 0
        for idx, item in enumerate(eval_dict['annotations']):
            eval_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1
        
        with EVAL_LABEL_PATH.open('w') as f:
            json.dump(eval_dict, f)
    else:
        print('Loading Json file...')


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
