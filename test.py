from src.models.backbone import ResNet10

model1 = ResNet10(include_top=True,
             input_shape=(224,224,3),
             pooling=None,
             classes=10)
print(model1.summary())

model2 = ResNet10(include_top=False,
             input_shape=(224,224,3),
             pooling=True,
             classes=10)
print(model2.summary())