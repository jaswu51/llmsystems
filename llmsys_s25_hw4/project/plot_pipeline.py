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
    # Read model parallel results
    mp_times = []
    mp_tokens = []
    
    mp_files = glob.glob('./workdir_model_parallel/eval_results_epoch*.json')
    for file in mp_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if 'training_time' in data and 'tokens_per_sec' in data:
                mp_times.append(data['training_time'])
                mp_tokens.append(data['tokens_per_sec'])
    
    # Read pipeline parallel results
    pp_times = []
    pp_tokens = []
    
    pp_files = glob.glob('./workdir_pipeline_parallel/eval_results_epoch*.json')
    for file in pp_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if 'training_time' in data and 'tokens_per_sec' in data:
                pp_times.append(data['training_time'])
                pp_tokens.append(data['tokens_per_sec'])
    
    # Calculate means and standard deviations
    mp_time_mean = np.mean(mp_times) if mp_times else None
    mp_time_std = np.std(mp_times) if mp_times else None
    mp_tokens_mean = np.mean(mp_tokens) if mp_tokens else None
    mp_tokens_std = np.std(mp_tokens) if mp_tokens else None
    
    pp_time_mean = np.mean(pp_times) if pp_times else None
    pp_time_std = np.std(pp_times) if pp_times else None
    pp_tokens_mean = np.mean(pp_tokens) if pp_tokens else None
    pp_tokens_std = np.std(pp_tokens) if pp_tokens else None
    
    # Print collected data
    print(f"Model Parallel training time: {mp_time_mean} ± {mp_time_std}")
    print(f"Pipeline Parallel training time: {pp_time_mean} ± {pp_time_std}")
    
    print(f"Model Parallel throughput: {mp_tokens_mean} ± {mp_tokens_std}")
    print(f"Pipeline Parallel throughput: {pp_tokens_mean} ± {pp_tokens_std}")
    
    # Plot training time chart
    means = []
    stds = []
    labels = []
    
    if pp_time_mean is not None:
        means.append(pp_time_mean)
        stds.append(pp_time_std if pp_time_std is not None else 0)
        labels.append('Pipeline Parallel')
    
    if mp_time_mean is not None:
        means.append(mp_time_mean)
        stds.append(mp_time_std if mp_time_std is not None else 0)
        labels.append('Model Parallel')
    
    if means:
        plot(means, stds, labels, 'submit_figures/ASSIGN_4_2_training_time.png', 'GPT2 Execution Time (Second)')
        print("Generated training time chart: submit_figures/ASSIGN_4_2_training_time.png")
    
    # Plot throughput chart
    tokens_means = []
    tokens_stds = []
    tokens_labels = []
    
    if pp_tokens_mean is not None:
        tokens_means.append(pp_tokens_mean)
        tokens_stds.append(pp_tokens_std if pp_tokens_std is not None else 0)
        tokens_labels.append('Pipeline Parallel')
    
    if mp_tokens_mean is not None:
        tokens_means.append(mp_tokens_mean)
        tokens_stds.append(mp_tokens_std if mp_tokens_std is not None else 0)
        tokens_labels.append('Model Parallel')
    
    if tokens_means:
        plot(tokens_means, tokens_stds, tokens_labels, 'submit_figures/ASSIGN_4_2_tokens_per_second.png', 'Tokens Per Second')
        print("Generated throughput chart: submit_figures/ASSIGN_4_2_tokens_per_second.png")
    
    # Calculate speedup
    if mp_time_mean is not None and pp_time_mean is not None:
        speedup = mp_time_mean / pp_time_mean
        print(f"Training time speedup: {speedup:.2f}x")
    
    if mp_tokens_mean is not None and pp_tokens_mean is not None:
        throughput_improvement = pp_tokens_mean / mp_tokens_mean
        print(f"Throughput improvement: {throughput_improvement:.2f}x") 