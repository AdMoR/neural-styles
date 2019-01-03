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

In another tab, you can run
`sudo docker run -d -p 6006:6006 -v $(pwd)/logs:/logs --name my-tf-tensorboard volnet/tensorflow-tensorboard`
It will allow you to visualize the loss of the current training in a tensorboard tab. Open `http://localhost:6006/#scalars` to get the visualization


## :gem: Some results :gem:
Some result on channel excitation `L[:, c, :, :]`: 

<<<<<<< HEAD
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
=======
On AlexNet, last convolution layer : 

![Example chanel excitation](images/LayerExcitationLoss_alexnet_1_34_2048_0.0005.jpg)
![Example chanel excitation 2](images/LayerExcitationLoss_alexnet_1_18_2048_0.0005.jpg)
![Example chanel excitation 3](images/LayerExcitationLoss_alexnet_1_15_2048_0.0005.jpg)

On VGG16, Conv 5_1 : 

![Example chanel excitation4](images/vgg16_conv_5_1-LayerExcitationLoss322+BatchDiversity-4-0.001-100-1024-0.jpg)
![Example chanel excitation5](images/vgg16_conv_5_1-LayerExcitationLoss396+BatchDiversity-4-0.001-100-1024-3.jpg)
![Example chanel excitation5](images/vgg16_conv_5_1-LayerExcitationLoss48+BatchDiversity-4-0.001-100-1024-3.jpg)

On ResNet18, fourth residual block : 

![Example chanel excitation4](images/resnet18_3-LayerExcitationLoss2+BatchDiversity-4-0.001-100-1024-2.jpg)
![Example chanel excitation5](images/resnet18_3-LayerExcitationLoss69+BatchDiversity-4-0.001-100-1024-3.jpg)
![Example chanel excitation5](images/resnet18_3-LayerExcitationLoss70+BatchDiversity-4-0.001-100-1024-1.jpg)
>>>>>>> 9f4ef65025b3212ddd6fc5ec82350a227ebd63ee

