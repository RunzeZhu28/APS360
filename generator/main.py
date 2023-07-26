import os
import music21 as m21
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

from lstm_module import LSTMModule
from data_processing import load_tokenized_songs, process_files
from model_training import train_model
import random


def main():
    random.seed(1)
    # Load and process the dataset
    dataset_path = "C:/Users/ASUS/Desktop/musicGPT/electronic"  # Update the dataset path accordingly
    desired_length = 100
    songs = load_tokenized_songs(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    folder_path = "C:/Users/ASUS/Desktop/musicGPT/converted_songs"  # Update the save folder path accordingly
    os.makedirs(folder_path, exist_ok=True)
    for i, song in enumerate(songs):
        converted_song = " ".join(map(str, song))
        save_path = os.path.join(folder_path,  str(i))
        with open(save_path, "w") as fp:
            fp.write(converted_song)

    # Process the dataset and perform one-hot encoding
    element_mapping = process_files(folder_path)
    encoded_arrays = one_hot_encoding(folder_path, element_mapping, desired_length)

    # Initialize the LSTM module
    input_size = len(element_mapping)
    hidden_size = 128
    output_size = len(element_mapping)
    dropout_prob = 0.2
    model = LSTMModule(input_size, hidden_size, output_size, dropout_prob)
    print(model)

    # Set hyperparameters and train the model
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    train_model(model, element_mapping,encoded_arrays, batch_size, num_epochs, learning_rate)

def one_hot_encoding(folder_path, element_mapping, desired_length):
    encoded_tensors = []  # List to store encoded tensors

    files = os.listdir(folder_path)
    files.sort()

    all_elements = set(element_mapping.values())

    for file in files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'r') as f:
            content = f.read().split()

        encoded_content = [element_mapping.get(element, 0) for element in content]
        encoded_content = torch.tensor(encoded_content)  # Convert to PyTorch tensor

        # Pad or truncate the tensor to the desired length
        if len(encoded_content) < desired_length:
            padded_content = nn.functional.pad(encoded_content, (0, desired_length - len(encoded_content)))
        else:
            padded_content = encoded_content[:desired_length]

        encoded_tensors.append(padded_content)

    # Pad sequences to the same length
    padded_sequences = pad_sequence(encoded_tensors, batch_first=True)

    encoded_arrays = []
    for tensor, file in zip(padded_sequences, files):
        
        # Convert tensor to numpy array
        onehot_encoded = np.zeros((tensor.size(0), len(all_elements)))
        for i, element in enumerate(tensor):
            if element.item() in element_mapping.values():
                onehot_encoded[i, element.item()] = 1

        encoded_arrays.append((onehot_encoded, file))

    return encoded_arrays

if __name__ == "__main__":
    main()

