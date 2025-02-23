from functools import partial
import time
import os
import fire
import tqdm
import json
import random
import datasets
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer
import pickle

import minitorch
from minitorch import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_functions import tensor_from_numpy



import subprocess
import os
 
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

def get_dataset(dataset_name, model_max_length):
    """
    Obtrain IWSLT (de-en) dataset.
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(
                example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]  # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key

def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Trains a tokenizer on the provided dataset examples and saves the tokenizer configuration.

    Parameters:
    - examples: The dataset examples used for training the tokenizer.
    - vocab_size: The desired vocabulary size for the tokenizer.
    - src_key: The key used to access the source text within the dataset examples.
    - tgt_key: The key used to access the target text within the dataset examples.
    - workdir: The directory where the tokenizer should be saved.

    Returns:
    - tokenizer: The trained tokenizer with special tokens,
        e.g., ("<eos_de>", "<eos_en>", "<pad>") if src_key and tgt_key are "de" and "en", respectively.
    """
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepares a batch of examples for model training or evaluation by tokenizing and padding them.

    Parameters:
    - examples: A list of examples to be processed.
    - src_key: The key for accessing source texts in the examples.
    - tgt_key: The key for accessing target texts in the examples.
    - tokenizer: The tokenizer to be used for encoding the texts.
    - model_max_length: The maximum sequence length the model can handle.
    - backend: The backend of minitorch tensors.

    Returns:
    - A dictionary containing keys: 'input_ids', 'labels', 'label_token_weights',
        each indicates a minitorch tensor with shape (len(examples), model_max_length).

    Notes:
    ["input_ids"] for every example in the DE-EN translation, the "input_ids" will be:
        <de_token_ids> + <de_eos_id> + <en_token_ids> + <en_eos_id> + <pad_ids>
    where the pad_ids makes the length of input_ids to be model_max_length.

    ["labels"]: the next tokens to be predicted, which will be used in the cross-entropy
    loss function, e.g., for an example tokenized as [a, b, c, d], "input_ids" and "labels" 
    can be [a, b, c] and [b, c, d], respectively.

    ["label_token_weights"] The 'label_token_weights' are used to differentiate
    calculation purposes. (the MLE loss is computed on target tokens only.)
    between the source (weight = 0) and target (weight = 1) tokens for loss
    """
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    # TODO: make examples in a 1d list, provide shape to initialize minitorch.Tensor
    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels    = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels    = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
    # input_ids = token_ids[:, :-1].tolist()
    # labels    = token_ids[:, 1:].tolist()
    # label_token_weights = tgt_token_mask[:, 1:].tolist()

    # input_ids = minitorch.tensor(input_ids, backend=backend)
    # labels    = minitorch.tensor(labels, backend=backend)
    # label_token_weights = minitorch.tensor(label_token_weights, backend=backend)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    The MLE loss for a batch.

    Parameters:
    - batch: The result of collate_fn, a dict with "input_ids", "labels", and "label_token_weights".
    - model: The model to be trained.

    Returns:
    - A scalar loss value for this batch, averaged across all target tokens.
    """

    idx = batch['input_ids']
    idx.requires_grad_(True)
    # print("getting into loss_fn")
    logits = model(idx=idx)
    # print("finish prediction")
    bs, l, c = logits.shape
    logits = logits.view(bs * l, c)
    targets = batch['labels'].view(bs * l)
    label_token_weights = batch['label_token_weights'].view(bs * l)

    targets.requires_grad_(True)
    # print("start calculating loss")
    # import pdb
    # pdb.set_trace()
    loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=targets
    )

    return ((loss * label_token_weights).sum() / label_token_weights.sum())


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    """
    Trains the model on the provided examples.

    Parameters:
    - model: The model to be trained.
    - optimizer: The optimizer used for updating the model's parameters.
    - examples: The dataset examples used for training.
    - n_samples: The random samples to train from "examples".
    - collate_fn: The function to collate data examples into batches.
    - batch_size: The number of examples in each batch.
    - desc: Description for the training process (used in progress bars).
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        t1 = time.time()

        loss.backward()
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        print(f"Forward: {t1 - t0}")
        print(f"Backward: {t2 - t1}")
        print(f"Opt.step: {t3 - t2}")

        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
            loss=loss.item(),
            lr=optimizer.lr)


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    """
    Evaluates the model on the provided examples and computes the average loss.

    Parameters:
    - model: The model to be evaluated.
    - examples: The dataset examples used for evaluation.
    - batch_size: The number of examples in each batch.
    - collate_fn: The function to collate data examples into batches.
    - desc: Description for the evaluation process (used in progress bars).

    Returns:
    - The average loss computed over all batches.
    """
    model.eval()
    losses = []

    for i in (prog_bar := tqdm.trange(
        0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])
        loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(model,
             examples,
             src_key,
             tgt_key,
             tokenizer,
             model_max_length,
             backend,
             desc):
    """
    Generates target sequences for the given source sequences using the model, based on argmax decoding.
    Note that it runs generation on examples one-by-one instead of in a batched manner.

    Parameters:
    - model: The model used for generation.
    - examples: The dataset examples containing source sequences.
    - src_key: The key for accessing source texts in the examples.
    - tgt_key: The key for accessing target texts in the examples.
    - tokenizer: The tokenizer used for encoding texts.
    - model_max_length: The maximum sequence length the model can handle.
    - backend: The backend of minitorch tensors.
    - desc: Description for the generation process (used in progress bars).

    Returns:
    - A list of generated target sequences.
    """

    model.eval()
    gen_sents = []
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        # Run generation for every single example

        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)


        while len(token_ids) <= model_max_length:

            token_ids_tensor = tensor_from_numpy(np.array([token_ids]), backend)
            
            # get logits
            logits = model(idx=token_ids_tensor)
            # logits of the last token
            logits_np = logits.to_numpy()[:, -1, :]

            # get the argmax
            gen_id = np.argmax(logits_np, axis=-1).item()

            if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    """
    Evaluates the BLEU score for generated sentences against the target sentences in the examples.

    Parameters:
    - examples: The dataset examples used for evaluation.
    - gen_sents: The generated sentences to be evaluated.
    - tgt_key: The key for accessing target texts in the examples.

    Returns:
    - A dictionary containing the BLEU score.
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }
def save_module_parameters(module, prefix=''):
    """Helper function to recursively save parameters of nested modules"""
    params = {}
    
    # Save direct parameters of this module
    for name, param in module.__dict__.items():
        if isinstance(param, minitorch.Parameter):
            params[f"{prefix}{name}"] = param.data.to_numpy()
            
    # Recursively save parameters of nested modules
    for name, child in module.__dict__.items():
        if isinstance(child, minitorch.Module):
            child_params = save_module_parameters(child, prefix=f"{prefix}{name}.")
            params.update(child_params)
            
    return params

def load_module_parameters(module, params_dict, backend):
    """Helper function to recursively load parameters into nested modules"""
    # Load direct parameters of this module
    for name, param in module.__dict__.items():
        if isinstance(param, minitorch.Parameter):
            if name in params_dict:
                param.data = tensor_from_numpy(params_dict[name], backend=backend)
                
    # Recursively load parameters of nested modules
    for name, child in module.__dict__.items():
        if isinstance(child, minitorch.Module):
            prefix = f"{name}."
            child_params = {
                k[len(prefix):]: v 
                for k, v in params_dict.items() 
                if k.startswith(prefix)
            }
            load_module_parameters(child, child_params, backend)

def save_checkpoint(epoch, model, optimizer, workdir):
    """
    Save training checkpoint including model parameters and optimizer state
    """
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model parameters recursively
    model_params = save_module_parameters(model)
    
    # Save optimizer state
    optimizer_state = {
        'lr': optimizer.lr,
        'beta1': optimizer.beta1,
        'beta2': optimizer.beta2,
        'eps': optimizer.eps,
        'states': {}
    }
    
    # Save optimizer states for each parameter
    for param_id, state in optimizer._states.items():
        if state:  # Only save non-empty states
            optimizer_state['states'][str(param_id)] = {
                'step': state['step'],
                'exp_avg': state['exp_avg'].to_numpy() if state['exp_avg'] is not None else None,
                'exp_avg_sq': state['exp_avg_sq'].to_numpy() if state['exp_avg_sq'] is not None else None
            }
    
    checkpoint = {
        'epoch': epoch,
        'model_params': model_params,
        'optimizer_state': optimizer_state
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint for epoch {epoch} at {checkpoint_path}")

def load_latest_checkpoint(workdir, model, optimizer):
    """Load the latest checkpoint with verification"""
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints found. Starting from scratch.")
        return 0
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        print("No checkpoints found. Starting from scratch.")
        return 0
        
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # 保存加载前的参数快照
        before_params = save_module_parameters(model)
        
        # 加载模型参数
        load_module_parameters(model, checkpoint['model_params'], model.backend)
        
        # 保存加载后的参数快照
        after_params = save_module_parameters(model)
        
        # 验证参数是否真的改变了
        params_changed = False
        for key in before_params.keys():
            if not np.array_equal(before_params[key], after_params[key]):
                params_changed = True
                break
                
        if not params_changed:
            print("WARNING: Model parameters did not change after loading checkpoint!")
            print("This might indicate a loading problem.")
        else:
            print("Successfully verified parameter changes after loading.")
        
        # 加载优化器状态
        optimizer.lr = checkpoint['optimizer_state']['lr']
        optimizer.beta1 = checkpoint['optimizer_state']['beta1']
        optimizer.beta2 = checkpoint['optimizer_state']['beta2']
        optimizer.eps = checkpoint['optimizer_state']['eps']
        print(f"Optimizer state loaded: lr={optimizer.lr}, beta1={optimizer.beta1}")
        
        # 恢复优化器状态
        states_loaded = 0
        for param_id_str, state in checkpoint['optimizer_state']['states'].items():
            param_id = int(param_id_str)
            if param_id in optimizer._states:
                optimizer._states[param_id] = {
                    'step': state['step'],
                    'exp_avg': tensor_from_numpy(state['exp_avg'], backend=model.backend) if state['exp_avg'] is not None else None,
                    'exp_avg_sq': tensor_from_numpy(state['exp_avg_sq'], backend=model.backend) if state['exp_avg_sq'] is not None else None
                }
                states_loaded += 1
        print(f"Loaded {states_loaded} optimizer states")
        
        next_epoch = checkpoint['epoch'] + 1
        print(f"Successfully resumed from epoch {checkpoint['epoch']}")
        return next_epoch
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print("Starting from scratch.")
        return 0
def main(dataset_name='bbaaaa/iwslt14-de-en-preprocess',
         model_max_length=40,
         n_epochs=20,
         batch_size=128,
         learning_rate=0.02,
         samples_per_epoch=200,
         n_vocab=10000,
         n_embd=256,
         seed=11111,
         checkpoint_frequency=1):

    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)

    config = {
        'n_vocab': n_vocab,
        'n_embd': n_embd,
        'n_head': 8,
        'n_positions': model_max_length,
        'p_dropout': 0.1,
        'ln_eps': 1e-5,
        'backend': backend
    }

    # 修改这里：不要直接 json.dumps 整个 config
    print("Initializing model with config:")
    print(f"- n_vocab: {config['n_vocab']}")
    print(f"- n_embd: {config['n_embd']}")
    print(f"- n_head: {config['n_head']}")
    print(f"- n_positions: {config['n_positions']}")
    print(f"- p_dropout: {config['p_dropout']}")
    print(f"- ln_eps: {config['ln_eps']}")
    print(f"- backend: {type(config['backend']).__name__}")  
    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    # Save initial model parameters for comparison
    print("Saving initial model parameters...")
    initial_params = save_module_parameters(model)
    
    # Load latest checkpoint if exists
    print("\nAttempting to load latest checkpoint...")
    start_epoch = load_latest_checkpoint(workdir, model, optimizer)
    
    # Verify parameter loading
    if start_epoch > 0:
        print("\nVerifying parameter changes after checkpoint loading...")
        current_params = save_module_parameters(model)
        params_changed = False
        unchanged_params = []
        for key in initial_params.keys():
            if np.array_equal(initial_params[key], current_params[key]):
                unchanged_params.append(key)
            else:
                params_changed = True
        
        if unchanged_params:
            print(f"WARNING: The following parameters did not change after loading checkpoint: {unchanged_params}")
        if not params_changed:
            print("CRITICAL WARNING: No parameters changed after loading checkpoint!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborting training...")
                return

    print("\nLoading dataset...")
    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)

    print("\nInitializing tokenizer...")
    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)

    print(f"\nStarting training from epoch {start_epoch}")
    best_validation_loss = float('inf')
    best_bleu_score = 0.0
    
    # Training loop
    for epoch_idx in range(start_epoch, n_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch_idx} / {n_epochs}")
        print(f"{'='*50}")
        
        desc = f'epoch {epoch_idx} / {n_epochs}'
        
        # Verify model parameters at the start of each epoch
        if epoch_idx > start_epoch:
            print("\nVerifying model parameters at epoch start...")
            current_params = save_module_parameters(model)
            unchanged_params = []
            for key in current_params.keys():
                if np.array_equal(current_params[key], initial_params[key]):
                    unchanged_params.append(key)
            if unchanged_params:
                print(f"WARNING: Following parameters haven't changed from initialization: {unchanged_params}")

        # Training
        print("\nStarting training phase...")
        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        # Validation
        print("\nStarting validation phase...")
        validation_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        print(f'\nEpoch {epoch_idx}: Validation Loss = {validation_loss}')
        
        # Track best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print(f"New best validation loss: {best_validation_loss}")

        # Generation and BLEU score evaluation
        print("\nGenerating translations for test set...")
        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            backend=backend,
            desc=desc)

        # Save generated examples
        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        
        gen_file = f'{workdir}/gen_epoch{epoch_idx}.json'
        print(f"\nSaving generated translations to {gen_file}")
        json.dump(gen_examples, open(gen_file, 'w'), indent=4)

        # Evaluate BLEU score
        print("\nCalculating BLEU score...")
        eval_scores = evaluate_bleu(
            examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
        print(f'Epoch {epoch_idx}: {eval_scores}')
        
        # Track best BLEU score
        current_bleu = eval_scores['bleu']
        if current_bleu > best_bleu_score:
            best_bleu_score = current_bleu
            print(f"New best BLEU score: {best_bleu_score}")

        # Save evaluation results
        eval_file = f'{workdir}/eval_results_epoch{epoch_idx}.json'
        print(f"\nSaving evaluation results to {eval_file}")
        json.dump(
            {
                'validation_loss': float(validation_loss),
                'best_validation_loss': float(best_validation_loss),
                'best_bleu_score': float(best_bleu_score),
                **eval_scores
            },
            open(eval_file, 'w'),
            indent=4
        )
            
        # Save checkpoint based on frequency
        if (epoch_idx + 1) % checkpoint_frequency == 0:
            print(f"\nSaving checkpoint for epoch {epoch_idx}...")
            save_checkpoint(epoch_idx, model, optimizer, workdir)

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    save_checkpoint(n_epochs - 1, model, optimizer, workdir)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_validation_loss}")
    print(f"Best BLEU score: {best_bleu_score}")


if __name__ == '__main__':
    fire.Fire(main)