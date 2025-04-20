import argparse
import os
import torch
from PIL import Image
from net import Generator_  
from utils import *

def main():
    # Initialize argument parser with description
    parser = argparse.ArgumentParser(description='Image fusion using PAFusion')
    
    parser.add_argument('--type', '-t', 
                       type=str,
                       default='lytro',
                       choices=['lytro', 'sice', 'tno', 'rs', 'pet', 'spect'],
                       help='Dataset type to process (default: pet)')
    
    args = parser.parse_args()

    model = Generator_().cuda()

    # Dataset configuration - path mappings
    dataset_config = {
        'lytro': {
            'A_path': "./test_img/lytro/far/",
            'B_path': "./test_img/lytro/near/",
            'weight_path': './models/lytro/DDAN5_G.pth'
        },
        'sice': {
            'A_path': "./test_img/SICE/over/",
            'B_path': "./test_img/SICE/under/",
            'weight_path': './models/sice/DDAN5_G.pth'
        },
        'tno': {
            'A_path': "./test_img/TNO/ir/",
            'B_path': "./test_img/TNO/vi/",
            'weight_path': './models/tno/DDAN5_G.pth'
        },
        'rs': {
            'A_path': "./test_img/RoadScene/ir/",
            'B_path': "./test_img/RoadScene/vi/",
            'weight_path': './models/roadscene/DDAN5_G.pth'
        },
        'pet': {
            'A_path': "./test_img/PET/mr/",
            'B_path': "./test_img/PET/pet/",
            'weight_path': './models/pet/DDAN5_G.pth'
        },
        'spect': {
            'A_path': "./test_img/SPECT/mr/",
            'B_path': "./test_img/SPECT/pet/",
            'weight_path': './models/spect/DDAN5_G.pth'
        }
    }

    if args.type not in dataset_config:
        raise ValueError(f"Invalid dataset type: {args.type}")
    
    print('Dataset type:',args.type)
    
    config = dataset_config[args.type]
    A_path = config['A_path']
    B_path = config['B_path']
    weight_path = config.get('weight_path', f'./models/{args.type}/DDAN5_G.pth')

    model.load_state_dict(torch.load(weight_path))
    model.eval()

    save_path = os.path.join('./results/', args.type)
    os.makedirs(save_path, exist_ok=True)


    with torch.no_grad():
        for root, _, files in os.walk(A_path):
            for filename in files:
                try:
                    img_A = image_loader(os.path.join(A_path, filename)).cuda()
                    img_B = image_loader(os.path.join(B_path, filename)).cuda()
                    
                    if img_A.shape != img_B.shape:
                        raise ValueError(f"Image dimension mismatch for {filename}")
                    
                    output = model(img_A, img_B)
                    output = denorm(output) * 255
                    img_np = output.squeeze(0).cpu().clone().numpy().clip(0, 255)
                    img_np = img_np.transpose(1, 2, 0).astype("uint8")
                    
                    img = Image.fromarray(img_np)
                    if args.type == 'tno':  # Convert to grayscale for TNO dataset
                        img = img.convert('L')
                    
                    output_filename = os.path.splitext(filename)[0] + '.png'
                    img.save(os.path.join(save_path, output_filename), 
                             compress_level=0)
                
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue


if __name__ == '__main__':
    main()
    print('Done')
