from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    # Generate schedules based on the expected pattern in the test cases
    # For m=3, n=3, the expected schedule is:
    # [(0,0)], [(1,0), (0,1)], [(2,0), (1,1), (0,2)], [(2,1), (1,2)], [(2,2)]
    
    # Total number of clock cycles
    total_cycles = num_batches + num_partitions - 1
    
    for clock in range(total_cycles):
        # List of activities for current clock cycle
        activities = []
        
        # For each partition, check if there's a microbatch to process
        for partition_idx in range(num_partitions):
            # Calculate which microbatch should be processed by current partition
            microbatch_idx = clock - partition_idx
            
            # If microbatch index is valid, add to activities
            if 0 <= microbatch_idx < num_batches:
                activities.append((microbatch_idx, partition_idx))
        
        yield activities
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN SOLUTION
        # Split input mini-batch into micro-batches
        micro_batches = []
        batch_size = x.size(0)
        for i in range(0, batch_size, self.split_size):
            micro_batches.append(x[i:i+self.split_size])
        
        num_micro_batches = len(micro_batches)
        num_partitions = len(self.partitions)
        
        # Generate clock schedule
        schedule = list(_clock_cycles(num_micro_batches, num_partitions))
        
        # Process each clock cycle
        for clock_activities in schedule:
            self.compute(micro_batches, clock_activities)
        
        # Get the last device
        last_device = self.devices[-1]
        
        # Concatenate micro-batches and move to the last device
        result = torch.cat([batch.to(last_device) for batch in micro_batches], dim=0)
        
        return result
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN SOLUTION
        for microbatch_idx, partition_idx in schedule:
            # Get the current partition
            partition = partitions[partition_idx]
            
            # Get the input and output queues for this partition
            in_queue = self.in_queues[partition_idx]
            out_queue = self.out_queues[partition_idx]
            
            # Get the current microbatch
            microbatch = batches[microbatch_idx]
            
            # Get the device for this partition
            device = devices[partition_idx]
            
            # Create a task to process the microbatch
            def compute_function():
                # Move the microbatch to the correct device
                microbatch_on_device = microbatch.to(device)
                # Process the microbatch
                return partition(microbatch_on_device)
            
            task = Task(compute_function)
            
            # Send the task to the worker
            in_queue.put(task)
            
            # Get the result from the worker
            success, result = out_queue.get()
            
            if not success:
                # If there was an error, raise the exception
                exc_info = result
                raise exc_info[1].with_traceback(exc_info[2])
            
            # Unpack the result
            _, output = result
            
            # Store the result back to batches
            batches[microbatch_idx] = output
        # END SOLUTION

