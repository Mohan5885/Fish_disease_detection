import timm

def build_model(model_name='resnet18', num_classes=2, pretrained=True):
    """
    Creates a classification model via timm.
    model_name examples: 'resnet18', 'efficientnet_b0', 'tf_efficientnet_b0'
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
