from src.utils import trainer
from src.models.custom_models import Baseline_plus
from src.utils import loader
from keras import optimizers

####################### training stage ###################
# load image
train_train_path = 'miniImageNet_category_split_train_phase_train.pickle'
train_val_path = 'miniImageNet_category_split_train_phase_val.pickle'

train_train = loader.load_miniImgnet(train_train_path)
train_val = loader.load_miniImgnet(train_val_path)

train_img = train_train[b'data']
train_label = train_train[b'labels']

val_img = train_val[b'data']
val_label = train_val[b'labels']

# define input shape, classes
input_shape = train_img.shape[1:]
classes = len(train_train[b'catname2label'])

# train and save weights
baseline_plus = Baseline_plus(input_shape=input_shape,classes=classes)
baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])
history_baseline_plus = trainer.train_model(model=baseline_plus,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=64,
                epochs=400 ,
                weights_file='baseline_plus_training',
                monitor='val_loss',
                )
