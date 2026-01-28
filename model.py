import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torch.optim as optim
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import copy
import itertools
from itertools import cycle

def load_dataset(train=True, download=True):
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    return dataset

class SynapticDownscalingModel(nn.Module):
    def __init__(self, p=0, nrem_replay=False):
        super(SynapticDownscalingModel, self).__init__()
        self.p = p
        self.nrem_replay = nrem_replay

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*8*8, 512)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        assert out.shape[2] == 16 and out.shape[3] == 16, f"Expected spatial dimensions to be 16x16, but got {out.shape[2]}x{out.shape[3]}"

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        assert out.shape[2] == 8 and out.shape[3] == 8, f"Expected spatial dimensions to be 8x8, but got {out.shape[2]}x{out.shape[3]}"

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc3(out)
        return out

def apply_percentile_downscaling(model, prune_percentile=0.2):    
    with torch.no_grad():
        for name, module in model.named_parameters():
            if isinstance(module, nn.Linear):
                param = module.weight
                
                tensor_flattened = param.abs().view(-1)
                
                threshold = torch.quantile(tensor_flattened, prune_percentile)

                mask = (param.abs() >= threshold).float()

                param.data.mul_(mask)

class ReplayBuffer(Dataset):
    def __init__(self, samples_per_class=200, mean=0.0, std=0.1):
        """
        Stores images and labels for replay.
        samples_per_class: How many real images to keep per specific class.
        """
        self.data = []
        self.targets = []
        self.samples_per_class = samples_per_class
        self.seen_classes = set()
        self.mean=mean
        self.std = std

    def add_data(self, dataset, class_labels):
        """
        Selects random samples from the provided dataset for the specific class_labels
        and adds them to the buffer.
        """
        for label in class_labels:
            if label in self.seen_classes:
                continue # We already have data for this class
            
            # Find all indices for this specific class
            all_targets = np.array(dataset.targets)
            indices = np.where(all_targets == label)[0]
            
            # Randomly select 'samples_per_class' indices
            if len(indices) > self.samples_per_class:
                selected_indices = np.random.choice(indices, self.samples_per_class, replace=False)
            else:
                selected_indices = indices # Take all if fewer than limit
            
            # Retrieve the actual images and store them
            for idx in selected_indices:
                img, target = dataset[idx] # dataset returns (Tensor, int)
                self.data.append(img)
                self.targets.append(target)
            
            self.seen_classes.add(label)
            print(f"   [Buffer] Added {len(selected_indices)} samples for Class {label}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_image_tensor = self.data[idx] + torch.randn(self.data[idx].size())*self.mean*self.std
        return noisy_image_tensor, self.targets[idx]
    
class UniformNoiseDataset(Dataset):
    """Dataset that generates random uniform noise."""
    def __init__(self, num_samples, noise_shape, low=0.0, high=1.0):
        """
        Args:
            num_samples (int): The total number of noise samples to generate.
            noise_shape (tuple): The shape of each individual noise sample (e.g., (3, 32, 32) for an image).
            low (float): The lower boundary of the output interval.
            high (float): The upper boundary of the output interval.
        """
        self.num_samples = num_samples
        self.noise_shape = noise_shape
        self.low = low
        self.high = high

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Generates and returns a single tensor of uniform noise."""
        # Generate random uniform noise using torch.rand, then scale it
        noise = self.low + (self.high - self.low) * torch.rand(self.noise_shape)
        
        # You can also use a target/label if needed (e.g., a dummy label of 0)
        label = torch.tensor(0) 
        
        return noise, label
    
def nrem_sleep(T, optimizer, model, teacher, replay_buffer, epochs=5, lr=1e-3, interleaved=False):
    replay_loader = DataLoader(replay_buffer, batch_size=64, shuffle=True)
    original = teacher
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
    if interleaved:
        epochs=5
    for epoch in range(epochs):
        print(f"NREM Replay Epoch {epoch+1}/{epochs}")
        model.train()
        original.eval()
        for param in original.parameters():
            param.requires_grad = False
        for images, _ in replay_loader:
            #images = images.to(next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(images)
            with torch.no_grad():
                target_outputs = original(images)
            loss = criterion(outputs, target_outputs) * (1 - 1/T)
            loss.backward()
            optimizer.step()

def train_model(model, train_dataset, test_dataset, epochs_per_task=10, batch_size=64, learning_rate=5e-4, p=0,
                synaptic_downscaling=False, pruning=False, nrem_replay=False, final_weights=None):
    
    criterion = nn.CrossEntropyLoss()
    distillation_criterion = nn.MSELoss()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_accuracies = []
    test_accuracies = []
    per_task_test_accuracies = [] 
    train_losses = []
    

    all_test_targets = np.array(test_dataset.targets)
    all_train_targets = np.array(train_dataset.targets)

    hippocampus = ReplayBuffer(samples_per_class=250)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for task in range(5):
        nrem_optimizer = optim.Adam(model.parameters(), lr=1e-4)
        print(f'\n=== Training on task {task+1}/5 ===')
        
        # 1. Prepare Training Data (Classes 2*task, 2*task+1)
        train_classes = [task*2, task*2+1]
        mask = np.isin(all_train_targets, train_classes)
        indices = np.where(mask)[0]
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        
        if task > 0 and nrem_replay==True:
            teacher = copy.deepcopy(model)
            teacher.eval()
            hippocampus_loader = DataLoader(hippocampus, batch_size=round(train_loader.batch_size*len(hippocampus.data)/len(train_loader.dataset)), shuffle=True)
        else:
            hippocampus_loader = DataLoader(UniformNoiseDataset(num_samples=500, noise_shape=(3,32,32), low=0.0, high=1.0), batch_size=round(train_loader.batch_size*500/len(train_loader.dataset)), shuffle=True)

        for epoch in range(epochs_per_task):
            print(f'Epoch {epoch+1}/{epochs_per_task}')
            
            #training
            model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for (images, labels), (replay_images, replay_labels) in zip(cycle(train_loader), (hippocampus_loader) ):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                if task > 0 and nrem_replay==True:
                    # noise = torch.randn_like(images)
                    teacher.eval()
                    teacher_outputs = teacher(replay_images)
                    model_outputs = model(replay_images)
                    distillation_loss = distillation_criterion(model_outputs, teacher_outputs)
                    ce_loss = criterion(outputs, labels)
                    #print(ce_loss.item(), distillation_loss.item())
                    loss = ce_loss + distillation_loss # ce_loss * (1/(task+1)) + distillation_loss * (1 - 1/(task+1))
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_correct += (outputs.argmax(1) == labels).sum().item()
                epoch_total += labels.size(0)
            
            train_acc = epoch_correct / epoch_total
            train_accuracies.append(train_acc)
            train_losses.append(epoch_loss/len(train_loader))

            #interleaved sleep phase
            # if nrem_replay and task > 0:
            #     nrem_sleep(task+1, nrem_optimizer, model, teacher, hippocampus, lr=1e-4, interleaved=True)

            if synaptic_downscaling and p > 0:
                apply_percentile_downscaling(model, prune_percentile=p)
            
            # --- Evaluation Phase ---
            model.eval()
            
            #calculate total test accuracy over ALL tasks seen so far
            total_correct_seen = 0
            total_samples_seen = 0
            
            
            current_epoch_task_accs = []
            
            with torch.no_grad():
                for t in range(0, 5):
                    
                    if t > task:
                        current_epoch_task_accs.append(0.0)
                        continue

                    task_classes = [t*2, t*2+1]
                    
                    test_mask = np.isin(all_test_targets, task_classes)
                    test_indices = np.where(test_mask)[0]
                    test_subset = Subset(test_dataset, test_indices)
                    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
                    
                    t_correct = 0
                    t_total = 0
                    
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        t_total += labels.size(0)
                        t_correct += (predicted == labels).sum().item()
                    
                    # Calculate per-task accuracy
                    task_acc = t_correct / t_total if t_total > 0 else 0.0
                    current_epoch_task_accs.append(task_acc)
                    
                    # Accumulate for the overall test accuracy
                    total_correct_seen += t_correct
                    total_samples_seen += t_total
            
            
            per_task_test_accuracies.append(current_epoch_task_accs)
            
            
            overall_test_acc = total_correct_seen / total_samples_seen if total_samples_seen > 0 else 0.0
            test_accuracies.append(overall_test_acc)
            
            print(f'  Train Acc: {train_acc:.4f} | Test Acc (Overall): {overall_test_acc:.4f}')
            print(f'  Per Task: {current_epoch_task_accs}')

            # intra task pruning
            # if pruning and p > 0:
            #     apply_percentile_downscaling(model, prune_percentile=p)
        
        #print(f"\n[Consolidating] Storing memories from Task {task}...")
        if nrem_replay:
            hippocampus.add_data(train_dataset, train_classes)

        # save model after task
        if p==0 and not nrem_replay:
            torch.save(model.state_dict(), f'models/model_after_task_{task}_no_downscaling.pth')
        
        # nrem sleep phase post train
        # if nrem_replay:
        #     nrem_sleep(model, hippocampus, epochs=5, lr=1e-4)

        
        # if synaptic_downscaling and p > 0:
        #     apply_percentile_downscaling(model, prune_percentile=p)

        # print out final layer weights
        if final_weights is not None:
            final_layer_weights = model.fc3.weight.data.cpu().numpy()
            final_weights.append((task, p, nrem_replay, final_layer_weights))
        

    return train_accuracies, test_accuracies, train_losses, per_task_test_accuracies

def plot_accuracies(train_accuracies, test_accuracies, epochs, task_epochs, model=None):
    plt.figure()
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    for i in range(5):
        plt.plot(epochs[i*task_epochs:task_epochs*i+task_epochs+1], train_accuracies[i*task_epochs:task_epochs*i+task_epochs+1], label=f'Train Accuracy for Task {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracies over Epochs')
    plt.legend()
    #plt.show()
    if model is not None:
        plt.savefig(f'results/train_test_accuracies___p_{model.p}_NREM_replay_{model.nrem_replay}.png')
    else:
        plt.savefig(f'results/train_test_accuracies_UNKNOWN.png')

def test_model(model, test_dataset, batch_size=64):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print(f'Test Accuracy: {test_acc:.4f}')
    return test_acc