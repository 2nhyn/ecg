#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
#!/usr/bin/env python

import numpy as np
import os
import sys
import random
import math
from helper_code import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
import wfdb
from util_nh import *

# ===================================================================

# fix seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# TRAIN FUNCTION

def train_model(data_folder, model_folder, verbose):
    # hyperparameter
    if verbose: print('Setting up hyperparameters and device...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRETRAIN_EPOCHS = 10
    FINETUNE_EPOCHS = 5
    PRETRAIN_BATCH_SIZE = 512
    FINETUNE_BATCH_SIZE = 64
    LR_PRETRAIN = 1e-3
    LR_FINETUNE = 1e-3
    MAX_LEN = 4096

    # data loading & preprocessing
    if verbose: print('Finding and preprocessing data...')
    records = find_records(data_folder)
    processor = SignalProcessor()
    
    signals_list = []
    labels_list = []
    sources_list = []

    if verbose:
        try:
            from tqdm import tqdm
            record_iterator = tqdm(records, desc="Preprocessing data")
        except ImportError:
            record_iterator = records
    else:
        record_iterator = records

    for record_path in record_iterator:
        full_path = os.path.join(data_folder, record_path)
        try:
            signal, meta = load_signals(full_path)
            header_string = load_header(full_path)
            label_val = get_label(header_string, allow_missing=True)
            source = get_source(header_string)

            if label_val is None or is_nan(label_val):
                if verbose: print(f"Skipping {record_path} due to missing label.")
                continue
            
            label = int(label_val)

            if np.isnan(signal).any():
                signal = np.nan_to_num(signal)

            processed_leads = []
            for i in range(signal.shape[1]):
                processed_lead = processor.preprocess(signal[:, i], meta['fs'])
                processed_leads.append(processed_lead)

            cleaned_signal = np.stack(processed_leads, axis=1)
            
            normalized_signal = normalize_leads(cleaned_signal)
            
            signals_list.append(normalized_signal)
            labels_list.append(label)
            sources_list.append(source)

        except Exception as e:
            if verbose: print(f"Skipping {record_path} due to error: {e}")
            continue

    if not signals_list:
        raise ValueError("No data could be processed. Please check the data folder and format.")

    padded_signals = pad_signals(signals_list, MAX_LEN)
    labels = np.array(labels_list)
    sources = np.array(sources_list)
    
    signals_tensor = torch.from_numpy(padded_signals).float().permute(0, 2, 1)

    # encoder pre-training
    if verbose: print('Starting self-supervised pre-training of the encoder...')
    encoder = ResNet1DEncoder(in_ch=padded_signals.shape[2]).to(device)
    pretrain_dataset = ContrastiveECGDataset(signals_tensor)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True)
    
    optimizer_pre = optim.AdamW(encoder.parameters(), lr=LR_PRETRAIN)
    criterion_pre = NTXentLoss().to(device)

    for epoch in range(PRETRAIN_EPOCHS):
        encoder.train()
        total_loss = 0
        for v1, v2 in pretrain_loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer_pre.zero_grad()
            _, z1 = encoder(v1)
            _, z2 = encoder(v2)
            loss = criterion_pre(z1, z2)
            loss.backward()
            optimizer_pre.step()
            total_loss += loss.item()
        if verbose: print(f'[Pre-train] Epoch {epoch+1}/{PRETRAIN_EPOCHS}, Loss: {total_loss/len(pretrain_loader):.4f}')

    # Linear Probing 
    if verbose: print('Starting supervised training of the classifier head...')
    
    # CODE15 제외
    finetune_mask = (sources != 'CODE-15%')
    finetune_signals_tensor = signals_tensor[finetune_mask]
    finetune_labels_array = labels[finetune_mask]

    if len(finetune_labels_array) == 0:
        if verbose: print("Warning: No non-CODE-15% data found for fine-tuning. Skipping fine-tuning.")
        head = LinearProbeHead(in_dim=512, num_classes=2)
    else:
        if verbose: print(f"Found {len(finetune_labels_array)} records from non-CODE-15% sources for fine-tuning.")
        
        head_save_path = os.path.join(model_folder, 'head.pth')
        trainer = LinearProbeTrainer(
            encoder=encoder,
            signals_tensor=finetune_signals_tensor,
            labels_array=finetune_labels_array,
            save_path=head_save_path,
            batch_size=FINETUNE_BATCH_SIZE,
            lr=LR_FINETUNE,
            device=device,
            verbose=verbose
        )
        trainer.train(epochs=FINETUNE_EPOCHS)
        head = trainer.head

    # final model save
    if verbose: print('Saving the final models...')
    os.makedirs(model_folder, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(model_folder, 'encoder.pth'))
    torch.save(head.state_dict(), os.path.join(model_folder, 'head.pth'))
    if verbose: print('Done.')


def load_model(model_folder, verbose):
    if verbose: print('Loading the models...')
    
    encoder = ResNet1DEncoder(in_ch=12)
    head = LinearProbeHead(in_dim=512, num_classes=2)
    
    encoder.load_state_dict(torch.load(os.path.join(model_folder, 'encoder.pth'), map_location=torch.device('cpu')))
    head.load_state_dict(torch.load(os.path.join(model_folder, 'head.pth'), map_location=torch.device('cpu')))
    
    model = {'encoder': encoder, 'head': head}
    
    return model

def run_model(record, model, verbose):

    encoder = model['encoder']
    head = model['head']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder.to(device)
    head.to(device)
    encoder.eval()
    head.eval()

    MAX_LEN = 4096
    processor = SignalProcessor()

    signal, meta = load_signals(record)
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal)
    
    processed_leads = []
    for i in range(signal.shape[1]):
        processed_lead = processor.preprocess(signal[:, i], meta['fs'])
        processed_leads.append(processed_lead)
    cleaned_signal = np.stack(processed_leads, axis=1)

    normalized_signal = normalize_leads(cleaned_signal)
    padded_signal = pad_signals([normalized_signal], MAX_LEN)
    signal_tensor = torch.from_numpy(padded_signal).float().permute(0, 2, 1).to(device)

    with torch.no_grad():
        h, _ = encoder(signal_tensor)
        logits = head(h)
        probabilities = F.softmax(logits, dim=1)
        
        probability = probabilities[0, 1].item()
        binary_prediction = 1 if probability >= 0.5 else 0

    return binary_prediction, probability

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)