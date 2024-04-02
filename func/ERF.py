import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import os
from utils.utils import attempt_load_weights


img_dir = ""
images = os.listdir(img_dir)
weight = ""
device = "cuda:0"
ckpt = torch.load(weight)
model = attempt_load_weights(weight, device)


model = model.eval()
input_H, input_W = 320, 320
heatmap = np.zeros([input_H, input_W])
print(model)


layer = model.model[8]
print(layer)


def farward_hook(data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)


for img in images:
    read_img = os.path.join(img_dir, img)
    image = Image.open(read_img)
    image = image.resize((input_H, input_W))
    image = np.float32(image) / 255
    input_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)

    input_tensor = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    input_tensor.requires_grad = True
    fmap_block = list()
    input_block = list()

    layer.register_forward_hook(farward_hook)
    output = model(input_tensor)
    feature_map = fmap_block[0].mean(dim=1, keepdim=False).squeeze()
    feature_map[(feature_map.shape[0] // 2 - 1)][(feature_map.shape[1] // 2 - 1)].backward(retain_graph=True)
    grad = torch.abs(input_tensor.grad)
    grad = grad.mean(dim=1, keepdim=False).squeeze()
    heatmap = heatmap + grad.cpu().numpy()

cam = heatmap
cam = cam / cam.max()
cam = 1 - cam
cam = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_SUMMER)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
cam_image = Image.fromarray(cam)


cam_image.save(f'')
print("save successful")