import os
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

# Real-ESRGAN 라이브러리를 불러옵니다 (이 부분은 Real-ESRGAN 설치에 따라 달라질 수 있습니다)
from realesrgan import RealESRGANer

# Real-ESRGAN 모델 초기화
# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
# model_path='/data/ephemeral/home/sr_model/RealESRGAN_x4plus.pth'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
model_path='/data/ephemeral/home/sr_model/RealESRGAN_x2plus.pth'

# IMAGE_ROOT = "/data/ephemeral/home/datasets/train/DCM"
IMAGE_ROOT = "/data/ephemeral/home/datasets/train/DCM"
SAVE_IMAGE_ROOT = "/data/ephemeral/home/datasets/train_resolx2x2/DCM"
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files  
    if os.path.splitext(fname)[1].lower() == ".png"
}
pngs = sorted(pngs)
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    dni_weight=None,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0)

for i in range(len(pngs)):
    image_path = os.path.join(IMAGE_ROOT, pngs[i])
    image = cv2.imread(image_path)

    output, _ = upsampler.enhance(image, outscale=4)

    save_image_path = os.path.join(SAVE_IMAGE_ROOT, pngs[i])
    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
    output = cv2.resize(output, dsize=(2048, 2048))
    cv2.imwrite(save_image_path, output)




print("Super Resolution 처리 완료!")