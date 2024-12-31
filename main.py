import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class MyTensor:
    def __init__(self) -> None:
        self.tensor = None


    def download_training_data(self):
        return datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )


    def download_test_data(self):
        return datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )


    def create_data_loaders(self):
        batch_size = 64
        train_data = self.download_training_data()
        test_data = self.download_test_data()
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break


    def get_tensor(self):
        return self.tensor


    def create_tensor_2(self):
        another_tensor = torch.ones(2, 2)
        sum_tensor = self.tensor + another_tensor
        print(f"sum of tensors: {sum_tensor}")


    # create a tensor
    def create_tensor(self):
        self.tensor = torch.rand(2, 2)
        print("Tensor: ")
        print(self.get_tensor())
        self.create_tensor_2()


def main():
    t = MyTensor()
    t.create_tensor()
    t_2 = MyTensor()
    t_2.create_tensor()
    t.create_data_loaders()
    t_2.create_data_loaders()
    x = complex(5, 4)
    print(f"x is : {x}")

if __name__ == "__main__":
    main()
