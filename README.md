## :wrench: INSTALLATION :wrench:
First install docker with nvidia support : [link](https://github.com/NVIDIA/nvidia-docker)

Go to the directory 
`cd neural-styles`

Start a docker with a pytorch image :

`sudo docker run --rm -it --init   --runtime=nvidia   --ipc=host   --user="$(id -u):$(id -g)"   --volume=$PWD:/app   -e NVIDIA_VISIBLE_DEVICES=0   anibali/pytorch /bin/bash`

Install the requirements
`pip install -r requirements.txt`

You can now run the script
`python3 neuron_excitation.py`


## :gem: Some results :gem:
Some result on channel excitation `L[:, c, :, :]`: 

On AlexNet : 

![Example chanel excitation](/images/LayerExcitationLoss_alexnet_1_34_2048_0.0005.jpg)
![Example chanel excitation 2](/images/LayerExcitationLoss_alexnet_1_18_2048_0.0005.jpg)
![Example chanel excitation 3](/images/LayerExcitationLoss_alexnet_1_15_2048_0.0005.jpg)

On VGG16 : 

![Example chanel excitation4](images/LayerExcitationLoss_vgg16_-1_4_2048_0.1_0.0005.jpg)
![Example chanel excitation5](images/LayerExcitationLoss_vgg16_-1_12_2048_0.1_0.0005.jpg)
![Example chanel excitation5](images/LayerExcitationLoss_vgg16_-1_10_2048_0.1_0.0005.jpg)

On VGG19 : 

![Example chanel excitation4](images/LayerExcitationLoss_vgg19_-1_15_2048_0.1_0.0005.jpg)

