## :wrench: INSTALLATION :wrench:

#### Classical neuron excitation
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


#### SVG neuron excitation
Executed on Linux 21.04
```
python3 -m venv venv
source venv
bash install_pydiffvg.sh
```


## :gem: Some results :gem:

#### Classical neuron excitation :accept:
Some result on channel excitation `L[:, c, :, :]`: 

On AlexNet, last convolution layer : 

![Example channel excitation](images/LayerExcitationLoss_alexnet_1_34_2048_0.0005.jpg)
![Example channel excitation 2](images/LayerExcitationLoss_alexnet_1_18_2048_0.0005.jpg)
![Example channel excitation 3](images/LayerExcitationLoss_alexnet_1_15_2048_0.0005.jpg)

On VGG16, Conv 5_1 : 

![Example channel excitation4](images/vgg16_conv_5_1-LayerExcitationLoss322+BatchDiversity-4-0.001-100-1024-0.jpg)
![Example channel excitation5](images/vgg16_conv_5_1-LayerExcitationLoss396+BatchDiversity-4-0.001-100-1024-3.jpg)
![Example channel excitation5](images/vgg16_conv_5_1-LayerExcitationLoss48+BatchDiversity-4-0.001-100-1024-3.jpg)

On ResNet18, fourth residual block : 

![Example channel excitation4](images/resnet18_3-LayerExcitationLoss2+BatchDiversity-4-0.001-100-1024-2.jpg)
![Example channel excitation5](images/resnet18_3-LayerExcitationLoss69+BatchDiversity-4-0.001-100-1024-3.jpg)
![Example channel excitation5](images/resnet18_3-LayerExcitationLoss70+BatchDiversity-4-0.001-100-1024-1.jpg)


#### SVG neuron excitation :pencil2:

B&W neuron optimization with lines only :penguin:

![Example neuron excitation svg](images/svg_neur_exc/result_n_paths200_im_size224_n_steps500_layer_nameVGGLayers.Conv4_3_layer_index1.svg)
![Example neuron oexcitation svg 2](images/svg_neur_exc/result_n_paths200_im_size224_n_steps500_layer_nameVGGLayers.Conv4_3_layer_index2.svg)


Color neuron optimization with lines only :rainbow:

![Example neuron excitation color svg](images/result_n_paths400_im_size224_n_steps1500_layer_nameVGGLayers.Conv4_3_layer_index5.svg)
![Example neuron excitation color svg 2](
images/result_n_paths400_im_size224_n_steps1500_layer_nameVGGLayers.Conv4_3_layer_index2.svg)



