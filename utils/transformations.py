import torchvision.transforms as transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size):
    transform_img = transforms.Compose( [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
    
    transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
    
    return transform_img, transform_mask
