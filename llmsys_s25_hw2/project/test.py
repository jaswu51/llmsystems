import pickle
import numpy as np

def inspect_checkpoint(checkpoint_path):
    """检查pkl文件中的所有参数信息"""
    print(f"\n检查检查点文件: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print("\n1. 基本信息:")
    print(f"- Epoch: {checkpoint['epoch']}")
    print(f"- 时间戳: {checkpoint['metadata']['timestamp']}")
    print(f"- Backend: {checkpoint['metadata']['backend']}")
    
    print("\n2. 优化器配置:")
    print(f"- Learning rate: {checkpoint['optimizer_state']['config']['lr']}")
    print(f"- Beta1: {checkpoint['optimizer_state']['config']['beta1']}")
    print(f"- Beta2: {checkpoint['optimizer_state']['config']['beta2']}")
    print(f"- Epsilon: {checkpoint['optimizer_state']['config']['eps']}")
    
    print("\n3. 模型参数:")
    for name, param in checkpoint['model_state_dict'].items():
        print(f"\n参数: {name}")
        print(f"- 形状: {param.shape}")
        print(f"- 类型: {param.dtype}")
        if isinstance(param, np.ndarray):
            print(f"- 值范围: [{param.min():.4f}, {param.max():.4f}]")
    
    print("\n4. 优化器状态:")
    for name, state in checkpoint['optimizer_state']['states'].items():
        print(f"\n状态: {name}")
        if 'exp_avg' in state:
            print(f"- exp_avg 形状: {state['exp_avg'].shape if state['exp_avg'] is not None else None}")
        if 'exp_avg_sq' in state:
            print(f"- exp_avg_sq 形状: {state['exp_avg_sq'].shape if state['exp_avg_sq'] is not None else None}")
        print(f"- step: {state['step']}")

# 使用示例:
checkpoint_path = "workdir_vocab10000_lr0.02_embd256/checkpoints/checkpoint_epoch_1.pkl"
inspect_checkpoint(checkpoint_path)