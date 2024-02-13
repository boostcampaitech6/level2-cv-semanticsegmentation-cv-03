import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import torch
import numpy as np
import streamlit as st
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision.models.segmentation import fcn_resnet50
import segmentation_models_pytorch as smp


IMAGE_ROOT = "/data/ephemeral/home/test/DCM"
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

images = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

title = []
for i, im in enumerate(images):
    title.append(im[:5])
title = list(set(title))
title = sorted(title)


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()
    

def gradcam(model, target_layers, index, image_path, hand):
    path = os.path.join(image_path, hand)

    image = Image.open(path).convert('RGB')
    image = image.resize((512,512))
    image = np.array(image)

    rgb_img = np.float32(image) / 255

    img = rgb_img.transpose(2, 0, 1)  
    img = torch.from_numpy(img).float()
    img = torch.Tensor(img)
    img = img.unsqueeze(0)

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = img.cuda()

    output = model(input_tensor)
    
    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu() 
    mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy() 
    mask_float = np.float32(mask==index)

    target = [SemanticSegmentationTarget(index, mask_float)]

    with GradCAM(model=model,
                target_layers=target_layers,) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0,:]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return cam_image


st.set_page_config(
    page_title = "GradCAM", 
    page_icon="📷", 
    layout = "wide", 
)

st.title("˙✧˖°📷 ༘ ⋆｡°")
st.markdown("<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #635985;'><br>", True)


model = smp.Unet(
    encoder_name = 'resnet34',
    encoder_weights = 'imagenet',
    in_channels = 3,
    classes = 29,
    activation = None
)

model = torch.load('/data/ephemeral/home/save_dir/resnet34_unet_best_model_start100.pt')
model = model.eval()

target_layers = [model.decoder.blocks[4].conv2]


img_name = st.selectbox("Image", title)
category = st.multiselect("Classes", CLASSES, default=CLASSES[0])    

right_cam, left_cam = [], []
for cate in category:
    index = CLASS2IND[cate]
    image_path = os.path.join(IMAGE_ROOT, img_name)
    image_file = os.listdir(image_path)

    right = image_file[0]
    left = image_file[1]

    right = gradcam(model, target_layers, index, image_path, right)
    left = gradcam(model, target_layers, index, image_path, left)

    right_cam.append(right)
    left_cam.append(left)


r_combined_cam = None
l_combined_cam = None

for cam in right_cam:
    if r_combined_cam is None:
        r_combined_cam = cam
    else:
        r_combined_cam = np.maximum(r_combined_cam, cam)

for cam in left_cam:
    if l_combined_cam is None:
        l_combined_cam = cam
    else:
        l_combined_cam = np.maximum(l_combined_cam, cam)

show_image_R = Image.fromarray(r_combined_cam)
show_image_L = Image.fromarray(l_combined_cam)


col1, col2 = st.columns(2)

with col1:
   st.header("Right")
   st.image(show_image_R)

with col2:
   st.header("Left")
   st.image(show_image_L)
