import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from model.newsan import san
from model.sanbaseline import sanbaseline
# Training Code
def setup_distributed_environment():
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        print("Doing parallel processing")
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        is_distributed = True
    else:
        local_rank = 0
        is_distributed = False
        if torch.cuda.is_available():
            print("Doing Single GPU Training")
            device = torch.device("cuda")
        else:
            print("Doing CPU training")
            device = torch.device("cpu")
    return device, local_rank, is_distributed

def main(sa_input):
    device, local_rank, is_distributed = setup_distributed_environment()

    num_classes = 10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10(root='/s/kabdulma/Workspace/SAN-master/data/', train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR10(root='/s/kabdulma/Workspace/SAN-master/data/', train=False, download=False, transform=transform_test)

    if is_distributed:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        test_sampler = DistributedSampler(testset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(trainset, batch_size=512, shuffle=(train_sampler is None), num_workers=16, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=16, pin_memory=True, sampler=test_sampler)

    # Initialize model. Make sure to replace `san` with the correct function or class, e.g., `SAN`.
    # 0 = pairwise 
    # 1 = patchwise
    
    # Set 'sa' based on the command-line argument
    global modeltype
    sa = 0 if sa_input == "pair" else 1
    modeltype = "Pairwise" if sa_input == "pair" else "Patchwise"
    
    if(sa == 0):
      print("The self attention model is initialized in pairwise")
    else:
      print("The self attention model is initialized in patchwise")
      
      
    model = san(attention_type=sa, layers=(2, 1, 2, 2), kernels=[3, 7, 7, 7], num_classes=num_classes, in_channels=3, dropout_rate=0.3).to(device)
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    epoch_losses = []
    epochs = 250
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # Aggregate loss from all processes if in a distributed environment
        if is_distributed:
            total_loss = torch.tensor([running_loss], device=device)
            dist.reduce(total_loss, dst=0)
            if local_rank == 0:  # Only the main process should output the loss
                epoch_loss = total_loss.item() / len(train_loader.dataset)
                epoch_losses.append(epoch_loss)
                print(f'{modeltype} Epoch: {epoch+1}, Loss: {epoch_loss:.4f}')
        else:
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_losses.append(epoch_loss)
            print(f'{modeltype} Epoch: {epoch+1}, Loss: {epoch_loss:.4f}')

        # Calculate Accuracy while running loop
        if epoch % 10 == 0 or epoch == 99:  # Example condition to run validation
            cal_accuracy(model, val_loader, device)
         
        t1 = "Loss Curve over Epochs for" + modeltype + "Learning"   
        s1 = "LossCurve" + modeltype + "ModifiedModel"
    if local_rank == 0: 
        plt.figure(figsize=(10,6))
        plt.plot(range(1,epochs+1), epoch_losses, marker = 'o', linestyle = '-', color = 'b')
        plt.title(t1)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(s1)
        plt.close()
            

def cal_accuracy(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Aggregate total and correct across all processes
    total = torch.tensor([total], dtype=torch.float, device=device)
    correct = torch.tensor([correct], dtype=torch.float, device=device)
    dist.reduce(total, dst=0)
    dist.reduce(correct, dst=0)
    
    # Convert tensors to Python scalars
    total_scalar = total.item()
    correct_scalar = correct.item()
    
    # Only print from process 0
    if dist.get_rank() == 0:
        print(f'{modeltype} Testing Accuracy: {100 * correct_scalar / total_scalar:.2f}%') 

if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Train a model with specified SA type.')
        parser.add_argument('sa_type', type=str, choices=['pair', 'patch'],
                            help='Specify SA type: pair or patch')
        args = parser.parse_args()

        main(args.sa_type)
