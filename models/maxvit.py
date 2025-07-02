import timm


def create_maxvit_nano_rw_256(num_classes=1):
    # Carregar modelo pré-treinado
    model = timm.create_model(
        'maxvit_nano_rw_256.sw_in1k', pretrained=True, num_classes=num_classes)

    return model
