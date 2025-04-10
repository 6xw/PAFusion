from torchvision import transforms
from PIL import Image
import torch

def image_loader(path):
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                                ])
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)


def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,3,1,1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1,3,1,1).cuda()

    return img.mul(std).add(mean).clip(0,1)