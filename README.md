## INSTALLATION
First install docker with nvidia support : [link](https://github.com/NVIDIA/nvidia-docker)

Go to the directory 
`cd neural-styles`

Start a docker with a pytorch image
`sudo docker run --rm -it --init   --runtime=nvidia   --ipc=host   --user="$(id -u):$(id -g)"   --volume=$PWD:/app   -e NVIDIA_VISIBLE_DEVICES=0   anibali/pytorch /bin/bash`

Install the requirements
`pip install -r requirements.txt`

You can now run the script
`python3 neuron_excitation.py`


## Some results
Some tests using jitter on AlexNet
![Example chanel excitation](/images/LayerExcitationLoss_alexnet_1_34_2048_0.0005.jpg)

