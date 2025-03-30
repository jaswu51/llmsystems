import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import json
import os
import glob

def plot(means, stds, labels, fig_name, ylabel):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Create directory for saving figures
os.makedirs('submit_figures', exist_ok=True)

# Fill the data points here
if __name__ == '__main__':
    # Load single GPU results
    single_gpu_times = []
    single_gpu_tokens = []
    
    single_files = glob.glob('./workdir_world_size_1/rank*_results_epoch*.json')
    for file in single_files:
        with open(file, 'r') as f:
            data = json.load(f)
            single_gpu_times.append(data['training_time'])
            single_gpu_tokens.append(data['tokens_per_sec'])
    
    # Load dual GPU results
    gpu0_times = []
    gpu0_tokens = []
    gpu1_times = []
    gpu1_tokens = []
    
    gpu0_files = glob.glob('./workdir_world_size_2/rank0_results_epoch*.json')
    for file in gpu0_files:
        with open(file, 'r') as f:
            data = json.load(f)
            gpu0_times.append(data['training_time'])
            gpu0_tokens.append(data['tokens_per_sec'])
    
    gpu1_files = glob.glob('./workdir_world_size_2/rank1_results_epoch*.json')
    for file in gpu1_files:
        with open(file, 'r') as f:
            data = json.load(f)
            gpu1_times.append(data['training_time'])
            gpu1_tokens.append(data['tokens_per_sec'])
    
    # Calculate means and standard deviations
    single_mean = np.mean(single_gpu_times) if single_gpu_times else None
    single_std = np.std(single_gpu_times) if single_gpu_times else None
    single_tokens_mean = np.mean(single_gpu_tokens) if single_gpu_tokens else None
    single_tokens_std = np.std(single_gpu_tokens) if single_gpu_tokens else None
    
    device0_mean = np.mean(gpu0_times) if gpu0_times else None
    device0_std = np.std(gpu0_times) if gpu0_times else None
    device0_tokens_mean = np.mean(gpu0_tokens) if gpu0_tokens else None
    device0_tokens_std = np.std(gpu0_tokens) if gpu0_tokens else None
    
    device1_mean = np.mean(gpu1_times) if gpu1_times else None
    device1_std = np.std(gpu1_times) if gpu1_times else None
    device1_tokens_mean = np.mean(gpu1_tokens) if gpu1_tokens else None
    device1_tokens_std = np.std(gpu1_tokens) if gpu1_tokens else None
    
    # Print collected data
    print(f"Single GPU training time: {single_mean} ± {single_std}")
    print(f"GPU0 training time: {device0_mean} ± {device0_std}")
    print(f"GPU1 training time: {device1_mean} ± {device1_std}")
    
    print(f"Single GPU throughput: {single_tokens_mean} ± {single_tokens_std}")
    print(f"GPU0 throughput: {device0_tokens_mean} ± {device0_tokens_std}")
    print(f"GPU1 throughput: {device1_tokens_mean} ± {device1_tokens_std}")
    
    # Plot training time chart
    means = []
    stds = []
    labels = []
    
    if device0_mean is not None:
        means.append(device0_mean)
        stds.append(device0_std if device0_std is not None else 0)
        labels.append('Data Parallel - GPU0')
    
    if device1_mean is not None:
        means.append(device1_mean)
        stds.append(device1_std if device1_std is not None else 0)
        labels.append('Data Parallel - GPU1')
    
    if single_mean is not None:
        means.append(single_mean)
        stds.append(single_std if single_std is not None else 0)
        labels.append('Single GPU')
    
    if means:
        plot(means, stds, labels, 'submit_figures/ASSIGN_4_1_training_time.png', 'GPT2 Execution Time (Second)')
        print("Generated training time chart: submit_figures/ASSIGN_4_1_training_time.png")
    
    # Plot throughput chart
    # For data parallel, we need to sum the throughput from both GPUs
    dp_tokens_mean = None
    dp_tokens_std = None
    
    if device0_tokens_mean is not None and device1_tokens_mean is not None:
        # Calculate total throughput for each epoch
        total_tokens = []
        for i in range(min(len(gpu0_tokens), len(gpu1_tokens))):
            total_tokens.append(gpu0_tokens[i] + gpu1_tokens[i])
        
        dp_tokens_mean = np.mean(total_tokens)
        dp_tokens_std = np.std(total_tokens)
    
    tokens_means = []
    tokens_stds = []
    tokens_labels = []
    
    if dp_tokens_mean is not None:
        tokens_means.append(dp_tokens_mean)
        tokens_stds.append(dp_tokens_std)
        tokens_labels.append('Data Parallel (2 GPUs)')
    
    if single_tokens_mean is not None:
        tokens_means.append(single_tokens_mean)
        tokens_stds.append(single_tokens_std if single_tokens_std is not None else 0)
        tokens_labels.append('Single GPU')
    
    if tokens_means:
        plot(tokens_means, tokens_stds, tokens_labels, 'submit_figures/ASSIGN_4_1_tokens_per_second.png', 'Tokens Per Second')
        print("Generated throughput chart: submit_figures/ASSIGN_4_1_tokens_per_second.png")
    
    # Calculate speedup
    if single_mean is not None and device0_mean is not None:
        speedup = single_mean / device0_mean
        print(f"Training time speedup: {speedup:.2f}x")
    
    if single_tokens_mean is not None and dp_tokens_mean is not None:
        throughput_improvement = dp_tokens_mean / single_tokens_mean
        print(f"Throughput improvement: {throughput_improvement:.2f}x")