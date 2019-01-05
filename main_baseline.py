from src.utils import trainer
from src.models.custom_models import Baseline_plus
from src.utils import loader


# resize the images and prepare the training data
train_train_path = 'miniImageNet_category_split_train_phase_train.pickle'
train_val_path = 'miniImageNet_category_split_train_phase_val.pickle'

train_train = loader.load_miniImgnet(train_train_path)
train_val = loader.load_miniImgnet(train_val_path)

train_img = train_train[b'data']
train_label = train_train[b'label']

val_img = train_val[b'data']
val_label = train_val[b'label']

baseline_plus = Baseline_plus(input_shape=(224,224,3),classes=5)

history_baseline = trainer.train_model(model=baseline_plus,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=16,
                epochs=400 ,
                weights_file='baseline_plus_training',
                monitor='val_loss',
                )