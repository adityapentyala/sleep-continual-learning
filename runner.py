import numpy as np 
from model import load_dataset, SynapticDownscalingModel, train_model, plot_accuracies, test_model
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_dataset = load_dataset(train=True)
    test_dataset = load_dataset(train=False)
    num_epochs_per_task = 2
    num_tasks = 5

    base_model = SynapticDownscalingModel()
    
    models = [SynapticDownscalingModel(),
              SynapticDownscalingModel(p=0.25),
              SynapticDownscalingModel(p=0.5),
              SynapticDownscalingModel(p=0.75),
              SynapticDownscalingModel(nrem_replay=True),
              SynapticDownscalingModel(p=0.25, nrem_replay=True),
              SynapticDownscalingModel(p=0.5, nrem_replay=True),
              SynapticDownscalingModel(p=0.75,nrem_replay=True)]
    
    results = [] # (train_accuracies, test_accuracies, train_losses, per_task_test_accuracies)
    final_weights = [] # (task, p, nrem, weights)
    
    for model in models:
        print(f"\n\n=== Training Model: p={model.p}, NREM Replay={model.nrem_replay} ===")
        model.to(device)
        train_accuracies, test_accuracies, train_losses, per_task_test_accuracies = train_model(
        model, train_dataset, test_dataset, epochs_per_task=num_epochs_per_task, nrem_replay=model.nrem_replay, 
        p=model.p, final_weights=final_weights, noise_train=False)

        for task in range(num_tasks):
            print(f"weights after task {task}{final_weights[-1][3].shape}:\n {final_weights[-1][3]}\n")

        results.append((train_accuracies, test_accuracies, train_losses, per_task_test_accuracies))
        plot_accuracies(train_accuracies, test_accuracies, 
                        range(1, num_epochs_per_task*num_tasks+1), num_epochs_per_task, model=model)
        
        
    colors = plt.cm.get_cmap('tab10', num_tasks)
    task_labels = [f"Task {t}" for t in range(num_tasks)]


    #model_tests = [per_task_test_accuracies,per_task_test_accuracies_p25, per_task_test_accuracies_p50, per_task_test_accuracies_p75]
    labels = ['No Downscaling', 'p=0.25', 'p=0.5', 'p=0.75']
    for i in range(len(results)):
        task_test_accuracies = [[], [], [], [], []]
        for j in range(num_tasks):
            for k in range(len(results[i][3])):
                task_test_accuracies[j].append(results[i][3][k][j]*0.2)
        task_test_accuracies = np.array(task_test_accuracies)
        task_test_accuracies[1][num_epochs_per_task:]+=task_test_accuracies[0][num_epochs_per_task:]
        task_test_accuracies[2][num_epochs_per_task*2:]+=task_test_accuracies[1][num_epochs_per_task*2:]
        task_test_accuracies[3][num_epochs_per_task*3:]+=task_test_accuracies[2][num_epochs_per_task*3:]
        task_test_accuracies[4][num_epochs_per_task*4:]+=task_test_accuracies[3][num_epochs_per_task*4:]
        #print(task_test_accuracies)
        plt.figure()
        for t in range(num_tasks):
            plt.plot(range(t*num_epochs_per_task+1, len(task_test_accuracies[t])+1), task_test_accuracies[t][t*num_epochs_per_task:], label=task_labels[t])
            plt.xlabel('Epochs')
            plt.ylim(bottom=0, top=0.8)
            plt.ylabel('Test Accuracy')
            plt.title(f'Per Task Test Accuracy, p={models[i].p}, NREM_Replay={models[i].nrem_replay}')
            plt.legend()
        #plt.show()
        plt.savefig(f'results/per_task_test_accuracy___p={models[i].p}_NREM_Replay={models[i].nrem_replay}.png')
