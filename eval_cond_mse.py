if __name__ == "__main__":
    import math
    import numpy as np
    import os
    import torch
    from PIL import Image
    from argparse import ArgumentParser
    from ddpm_torch import *
    from torch.utils.data import Dataset, Subset, DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    parser = ArgumentParser()

    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--dataset", choices=DATASET_DICT.keys(), default="cifar10")
    parser.add_argument("--eval-batch-size", default=512, type=int)
    parser.add_argument("--eval-total-size", default=10000, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--sample-folder", default="", type=str)
    parser.add_argument("--sample-y-folder", default="", type=str)
    parser.add_argument("--num-gpus", default=1, type=int)

    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    dataset = args.dataset
    print(f"Dataset: {dataset}")

    folder_name = os.path.basename(args.sample_folder.rstrip(r"\/"))
    if args.num_gpus > 1:
        raise NotImplementedError
        # assert torch.cuda.is_available() and torch.cuda.device_count() >= args.num_gpus
        # model_device = [f"cuda:{i}" for i in range(args.num_gpus)]
        # input_device = "cpu"  # nn.DataParallel is input device agnostic
        # op_device = model_device[0]
    else:
        op_device = input_device = model_device = torch.device(args.device)

    args = parser.parse_args()

    eval_batch_size = args.eval_batch_size
    eval_total_size = args.eval_total_size
    num_workers = args.num_workers

    class ImageFolder(Dataset):
        def __init__(self, img_dir, transform=transforms.PILToTensor()):
            self.img_dir = img_dir
            self.img_list = sorted([
                img for img in os.listdir(img_dir)
                if img.split(".")[-1] in {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}])
            self.transform = transform

        def __getitem__(self, idx):
            with Image.open(os.path.join(self.img_dir, self.img_list[idx])) as im:
                return self.transform(im)

        def __len__(self):
            return len(self.img_list)

    imagefolder = ImageFolder(args.sample_folder)
    yfolder = ImageFolder(args.sample_y_folder)
    if len(imagefolder) > eval_total_size:
        inds = torch.as_tensor(np.random.choice(len(imagefolder), size=eval_total_size, replace=False))
        imagefolder = Subset(imagefolder, indices=inds)
        yfolder = Subset(yfolder, indices=inds)
    imageloader = DataLoader(
        imagefolder, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    yloader = DataLoader(
        yfolder, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    running_loss = 0.
    for x, x0 in tqdm(zip(imageloader, yloader), total=len(imageloader)):
        running_loss += torch.sum((x-x0).pow(2))
    running_loss /= eval_total_size
    print(running_loss)
