import timm


def create_inception_resnet_v2(num_classes=1):
    # Carregar modelo pré-treinado
    model = timm.create_model('inception_resnet_v2',
                              pretrained=True, num_classes=num_classes)

    return model
