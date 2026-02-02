"""
Noise Inference Script
parameters:
    test: bool - whether to test on test dataset or noise inputs
    ressurect: bool - whether to ressurect previous task weights
change nrem replay and p values in the model list as needed
"""
import torch
import numpy as np
from model import SynapticDownscalingModel, test_model, load_dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    high, low = 1.0, 0.0
    test = True
    ressurect = True
    noise_inputs = low + (high - low) * torch.rand((100, 3, 32, 32))
    test_data = load_dataset(train=False)
    models = [SynapticDownscalingModel(p=0, nrem_replay=False)]*5
    for task, model in enumerate(models):
        print(f"\n\n=== Evaluating Model after Task {task+1} ===")
        model.load_state_dict(torch.load(f"./models/model_after_task_{task}_no_downscaling.pth", map_location=device))
        model.eval()
        model.to(device)
        if test:
            if ressurect and task > 0:
                prev_model = SynapticDownscalingModel(p=0, nrem_replay=False)
                #prev_model.load_state_dict(torch.load(f"./models/model_after_task_{task-1}_no_downscaling.pth", map_location=device))
                
                #print("prev model weights", prev_output_weights.shape, prev_output_weights.type())
                for previous_task in range(task):
                    prev_model.load_state_dict(torch.load(f"./models/model_after_task_{previous_task}_no_downscaling.pth", map_location=device))
                    prev_model.to(device)
                    prev_output_weights = prev_model.fc3.weight.data.clone()
                    model.fc3.weight.data[previous_task*2] = prev_output_weights[previous_task*2]
                    model.fc3.weight.data[previous_task*2 + 1] = prev_output_weights[previous_task*2 + 1]
                print("Resurrected weights from previous model for previous tasks.")
            test_accuracy, unique, counts, distribution = test_model(model, test_data)
            print(f"Test Accuracy after Task {task+1}: {test_accuracy*100:.2f}%")
            print(f"Test Set Prediction Distribution after Task {task+1}: {distribution}")
        else:
            with torch.no_grad():
                inputs = noise_inputs[:].to(device)
                print(inputs.shape)
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                unique, counts = torch.unique(predicted, return_counts=True)
                distribution = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
                print(f"Task {task+1} checkpoint - Noise Input Prediction Distribution: {distribution}")
        plt.figure()
        plt.bar(distribution.keys(), distribution.values())
        plt.xlim(0, 9)
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title(f'Noise Input Prediction Distribution after Task {task+1}, NREM Replay={model.nrem_replay}, p={model.p}, Ressurect={ressurect}')
        plt.savefig(f'results/noise/test_data_preds_distribution_task_{task+1}_nrem={model.nrem_replay}_resurrect={ressurect}.png')

