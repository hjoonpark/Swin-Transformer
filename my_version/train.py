
from dataset import Dataset
import argparse
import torch
import sys
sys.path.append("..")
from config import get_config
from models import build_model

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', default="../configs/swin/swin_base_patch4_window7_224_22k.yaml")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU", default=1)
    parser.add_argument('--epochs', type=int, help="batch size for single GPU", default=10)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+', )
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel', default=0)
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config
    
if __name__ == "__main__":
    args, config = parse_option()
    print(args)
    # print(config)
    model_type = config.MODEL.TYPE
    print(model_type)
    device = torch.device("cuda:0")

    dataset_train = Dataset("train", config)
    dataset_val = Dataset("val", config)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    model = build_model(config)
    model = model.to(device)
    print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters:,}")
    model.train()

    for epoch in range(args.epochs):
        for idx, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(device)
            targets = targets.to(device)

            print(idx, samples.shape, targets.shape)
            outputs = model(samples)
            print(outputs.shape)
            break
        break
    print("$$$$$ DONE $$$$$")