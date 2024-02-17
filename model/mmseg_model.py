from mmseg.apis import init_model
from mmengine import Config
from base import BaseModel
import torch.nn.functional as F

def update_value(d, target_key, new_value):
    """
    재귀적으로 딕셔너리를 순회하여 특정 키를 찾고 값을 변경하는 함수
    Args:
        d (dict): 변경할 딕셔너리
        target_key (str): 변경하려는 값의 키
        new_value: 새로운 값
    """
    for key, value in d.items():
        if key == target_key:
            d[key] = new_value
        elif isinstance(value, dict):
            update_value(value, target_key, new_value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    update_value(item, target_key, new_value)

class MmsegModel(BaseModel):
    def __init__(
        self,
        num_classes=29,
        mmseg_config=None,
        pretrained= None


    ):
        super().__init__()
        assert mmseg_config is not None, 'enter the mmsegmentation config path into config.json.'
        config = Config.fromfile(mmseg_config)
        update_value(config, 'num_classes', num_classes)
        self.pretrained_model = init_model(config, pretrained)

    def forward(self, x):
        out = self.pretrained_model(x)
        out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode="bilinear")
        return out

