import torch.fx as fx
import torchvision.models as models

model = models.resnet50(weights='IMAGENET1K_V1')
# print(model)
model.eval()

traced = fx.symbolic_trace(model)
print(traced.graph)

