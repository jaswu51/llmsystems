#!/bin/bash

# 设置运行次数，如果没有指定，默认运行10次
num_runs=${1:-8}

# # 激活conda环境
# source ~/.bashrc
# conda activate minitorch

# 当前运行次数计数器
count=1

while [ $count -le $num_runs ]
do
    echo "Starting run $count of $num_runs"
    echo "--------------------------------"
    
    # 运行Python脚本
    python project/run_machine_translation.py
    
    # 检查上一个命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "Run $count completed successfully"
    else
        echo "Run $count failed"
    fi
    
    # 如果不是最后一次运行，则等待10秒
    if [ $count -lt $num_runs ]; then
        echo "Waiting 10 seconds before next run..."
        sleep 10
    fi
    
    # 增加计数器
    ((count++))
    
    echo "--------------------------------"
done

echo "All runs completed"