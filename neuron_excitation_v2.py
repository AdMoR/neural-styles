import torch

from neural_styles.optimizer_classes.visu_optimier_v2 import NeuronVisualizer
from neural_styles.optimizer_classes.objective_functions import ChannelObjective, ClipImageTextMatching
from neural_styles.optimizer_classes.generators import FourierGenerator
from neural_styles.nn_utils.prepare_model import VGG16Layers

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


gen = FourierGenerator()
obj = ChannelObjective(VGG16Layers.Conv4_3, 0)
reg = ClipImageTextMatching(lambda_reg=100.0)
#reg = None

print(gen, obj, reg)

optimizer = NeuronVisualizer(gen, obj, reg)

optimizer.gen('./')
