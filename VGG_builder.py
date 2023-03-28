import torch.nn as nn 
import torchvision
from cnn_workflow.utils import add_model_note

class VGG_builder:
    """
        Class to build VGG net with custom classifier
    """

    def  __init__(self, num_classes, **kwargs):
        """
        kwagrs:
            vgg_str can be:  vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19 .... see torchvision/models/vgg.py
        """
        self.num_classes = num_classes
        self.vgg_str = kwargs.get('vgg_str', None)
        self.classifier_idx = kwargs.get('classifier_idx', None)
        self.freeze_features = kwargs.get('freeze_features', False)
        self.model_note = kwargs.get('model_note', None)

    def get_classifier(self, in_features, idx):
        classifiers = [
            nn.Sequential(
                nn.Linear(in_features = in_features, out_features=4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096,self.num_classes)
            ),
            nn.Sequential(
                nn.Linear(in_features = in_features, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features = 4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_classes)
            ),
            nn.Sequential(
                nn.Linear(in_features = in_features, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features = 4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features = 4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_classes)
            )
        ]
        return classifiers[idx]

    def set_freeze_features(self, freeze):
        self.freeze_features = freeze

    @staticmethod
    def freeze_model_parameters(model):
        for param in model.parameters():
                param.requires_grad = False
    
    def classifier_str(self):
        return f'classifier {self.classifier_idx}'
    
    def build(self):
        vgg_fn = getattr(torchvision.models, self.vgg_str)
        model = vgg_fn(pretrained=True)
        add_model_note(model, self.vgg_str)
        if self.model_note:
            add_model_note(model, self.model_note)

        if self.freeze_features:
            self.freeze_model_parameters(model)

        in_features = model.classifier[0].in_features

        model.classifier = self.get_classifier(in_features, self.classifier_idx)
        add_model_note(model, self.classifier_str())

        return model