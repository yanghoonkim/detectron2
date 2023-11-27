FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git vim tmux htop python3-pip python3-dev ninja-build ffmpeg libsm6 libxext6 wget

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install opencv-python timm shapely~=1.0 pandas notebook==6.4.8 traitlets==5.9.0 polars

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /home

RUN git clone https://github.com/yanghoonkim/detectron2.git

WORKDIR /home/detectron2

RUN git checkout nia

RUN wget https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl -P nia/

ARG TORCH_CUDA_ARCH_LIST=8.6

RUN python -m pip install -e .

CMD [ "/bin/bash" ]