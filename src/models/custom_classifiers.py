from keras import backend
from src.models.custom_layers import CosDense

def cos_sim_classifier(feature_extracted,classes,activation='softmax'):
    """
    :param feature_extraxted: the input tensor
    :param n_classes:
    :return: the output tensor
    """
    x = CosDense(classes,activation=activation, name='cosine_dense')(feature_extracted)

    return x
