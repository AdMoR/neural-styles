import torch

from neural_styles.optimizer_classes.visu_optimier_v2 import NeuronVisualizer
from neural_styles.optimizer_classes.objective_functions import ChannelObjective, ClipImageTextMatching, DualMirrorLoss
from neural_styles.optimizer_classes.generators import FourierGenerator
from neural_styles.nn_utils.prepare_model import VGG16Layers

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


gen = FourierGenerator(size=224, sd=0.00001)
obj = DualMirrorLoss("A volcano", "Cats", 0.01, 0.01)
reg = None #ClipImageTextMatching(lambda_reg=100.0)
#reg = None

print(gen, obj, reg)

optimizer = NeuronVisualizer(gen, obj, reg, n_steps=10000, n_augment=4, lr=0.0001)

optimizer.gen('./')
