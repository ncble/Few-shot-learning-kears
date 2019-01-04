from keras import backend
from keras import layers
from keras import models
from src.models.backbones import ResNet10
from src.models.custom_classifiers import cos_sim_classifier

def baseline_plus(input_shape=None,classes=None):
    # input tensor
    img_input = layers.Input(shape=input_shape)
    # backbone
    x = ResNet10(img_input,
             include_top=False,
             pooling='avg',
             classes=None)
    #classifier based on cosine similarity
    x = cos_sim_classifier(x,classes)
    inputs = img_input
    model = models.Model(inputs, x, name='baseline_plus')
    return model

