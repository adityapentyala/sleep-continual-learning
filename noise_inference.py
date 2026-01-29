import torch
import numpy as np
from model import SynapticDownscalingModel, UniformNoiseDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_inputs = UniformNoiseDataset(num_samples=1000, noise_shape=(3, 32, 32))
    models = [SynapticDownscalingModel(p=0, nrem_replay=False)]*5
    for task, model in enumerate(models):
        print(f"\n\n=== Evaluating Model after Task {task+1} ===")
        model.load_state_dict(torch.load(f"models/model_after_task_{task}_no_downscaling.pth"))
        model.eval()
        model.to(device)
        with torch.no_grad():
            inputs = noise_inputs[:][0].to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            unique, counts = torch.unique(predicted, return_counts=True)
            distribution = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
            print(f"Task {task+1} checkpoint - Noise Input Prediction Distribution: {distribution}")
        plt.figure()
        plt.bar(distribution.keys(), distribution.values())
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title(f'Noise Input Prediction Distribution after Task {task+1}')
        plt.savefig(f'results/noise/noise_distribution_task_{task+1}.png')

