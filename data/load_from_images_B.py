import torchvision
import torch


def rgb_to_gray(t):
    # t.shape (3, d, d)
    Y = 0.299 * t[0, :, :] + 0.587 * t[1, :, :] + 0.114 * t[2, :, :]
    return Y


class ImageFolder(torchvision.datasets.ImageFolder):
    downsample = 1

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.downsample > 1:
            img = img[:, ::self.downsample, ::self.downsample]
        return img, target


def get_data_loader(batch_size, name, downsample=1, is_train=True):
    folder_name = 'trainB' if is_train else 'testB'
    imagenet_data = ImageFolder(
        'datasets/{}/{}'.format(name, folder_name),
        transform=torchvision.transforms.ToTensor())
    imagenet_data.downsample = downsample
    data_loader = torch.utils.data.DataLoader(
        imagenet_data, batch_size=batch_size, shuffle=True)
    return data_loader


