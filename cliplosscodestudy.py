# Contrastive Loss
import torch
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn
from open_clip.loss import ClipLoss

# 이미지와 text는 모두 배치 사이즈 16 그리고 1024차원으로 임베딩된다. 
#아래 코드는 개괄적인 loss의 진행을 나타낸 것이다. 
image_features = torch.randn(16, 1024)
text_features  = torch.randn(16, 1024)
loss_fn        = ClipLoss()
logit_scale    = nn.Parameter(torch.tensor(np.log(1/0.07)))
#여기서 logit scale은 초기값을 0.07로 설정하였고, 그 이유는 (Veeling et al. (2018)) 가 제안.
# 또한 이 파라미터는 모델 학습이 불안정해질 수 있기에 clipped to prevent scaling the logits by more than 100
loss = loss_fn(image_features, text_features, logit_scale)

# contrastive loss는 코사인 유사도를 대각행렬에 대해서는 최대화하고 그 이외는 최소화하는 시도를 의미한다. 
#아래는 psuedo 코드이다. 
labels = np.arange(n) 
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1) 
loss = (loss_i + loss_t)/2

#아래는 본격적으로 clip loss를 구현한 것이다. 
class ClipLoss(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels #라벨을 할당한다. 

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T #이미지 별 텍스트들에 대한 유사도를 보여줌. 
        logits_per_text = logit_scale * text_features @ image_features.T # 텍스트 별 이미지들에 대한 유사도를 보여줌.         
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2  #각각의 cross entropy loss를 평균 내면 됨. 
        return {"contrastive_loss": total_loss}
