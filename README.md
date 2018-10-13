`sudo docker run --rm -it --init   --runtime=nvidia   --ipc=host   --user="$(id -u):$(id -g)"   --volume=$PWD:/app   -e NVIDIA_VISIBLE_DEVICES=0   anibali/pytorch /bin/bash`
