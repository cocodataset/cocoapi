# docker build -t cocoapi .

# xhost +local:docker && docker run --rm -e "DISPLAY=${DISPLAY}" --ipc=host -it --gpus all -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /drive/cocoapi/:/home/dh/ -p 8888:8888 cocoapi

# jupyter notebook --ip=0.0.0.0 --allow-root

FROM nvcr.io/nvidia/pytorch:19.04-py3

RUN apt-get update
RUN apt-get install -qqy x11-apps
RUN apt-get install -y eog #for viewing image files
RUN pip install --upgrade pip
RUN pip install scikit-image
RUN pip install docopt visdom easydict tensorboardX json_tricks
RUN pip uninstall -y pycocotools
RUN python -m pip install pycocotools==2.0.0
RUN sed -i "s#debugName#uniqueName#g" /opt/conda/lib/python3.6/site-packages/tensorboardX/pytorch_graph.py
ENV DISPLAY :0

RUN mkdir -p /home/dh
WORKDIR /home/dh
COPY . /home/dh/
WORKDIR /home/dh/PythonAPI/
RUN make
WORKDIR /home/dh