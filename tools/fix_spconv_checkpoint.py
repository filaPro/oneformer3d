import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.in_path)
    key = 'state_dict'  # 'model' for SSTNet
    for layer in checkpoint[key]:
        if (layer.startswith('unet') or layer.startswith('input_conv')) \
            and layer.endswith('weight') \
                and len(checkpoint[key][layer].shape) == 5:
            checkpoint[key][layer] = checkpoint[key][layer].permute(1, 2, 3, 4, 0)
    torch.save(checkpoint, args.out_path)
