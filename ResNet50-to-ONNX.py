import torch, torchvision.models as models
model = models.resnet50(weights='IMAGENET1K_V1').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "resnet50.onnx", opset_version=12)
