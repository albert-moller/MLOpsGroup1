import torch
import timm
from torch import nn

class Model(nn.Module):
    """
    Implementation of the MobileNetV3 model
    using a pre-trained model from the TIMM 
    framework.
    """
    def __init__(self, model_name: str = 'mobilenetv3_small_100', num_classes: int = 38, pretrained: bool = True) -> None:
        """
        Initializes the MobileNetV3 Model.

        Args:
            model_name (str): Name of the pre-trained model from the TIMM framework.
            num_classes (int): Number of output classes for classification.
            pretrained (bool): Whether to load pre-trained weights.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = self.load_model()
        self.prepare_model_for_finetuning()

    def load_model(self) -> torch.nn.Module:
        """
        Initializes the MobileNetV3 Model using the TIMM framework.
        """
        model = timm.create_model(model_name=self.model_name, pretrained=self.pretrained)
        return model

    def prepare_model_for_finetuning(self) -> None:
        """
        Freezes all layers except the output classification layer. 
        Replaces the classification layer of the pre-trained
        model such that it accounts for the correct number
        of classes.
        """
        # Freeze all layers. 
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the last convolutional layer for fine-tuning.
        for param in self.model.blocks[-1].parameters():
            param.requires_grad = True
        # Replace the output classification layer.
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            # Unfreeze the linear output layer.
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.num_classes)
            # Unfreeze the classification layer.
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"The selected model {self.model_name} does not have an output classification layer.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.      
        """
        return self.model(x)
    
if __name__ == "__main__":
    # Initialize model for fine-tuning.
    model = Model()
    # Calculate the total number of model parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")   
    # Calculate the number of trainable parameters.
    finetune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters being fine-tuned: {finetune_params}")
    # Dummy input for testing.
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    