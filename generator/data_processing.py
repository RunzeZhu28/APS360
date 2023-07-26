import os
import music21 as m21
import torch
import torch.nn as nn
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np

def tokenize_score(score, time_step=0.25):
    tokens = []
    for part in score.parts:
        part_stream = part.flatten().notesAndRests
        prev_event = None
        symbol = -1  # Default value for 'symbol' variable

        for event in part_stream:
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi
            elif isinstance(event, m21.note.Rest):
                symbol = -1  # Use -1 to represent rests

            steps = int(event.duration.quarterLength / time_step)
            tokens.extend([symbol] + [-1] * (steps - 1))

    return tokens



def load_tokenized_songs(dataset_path, segment_length=100, step_size=10, max_neg_ones=30):
    songs = []
    for path, _, files in os.walk(dataset_path):
        for file in files:
            song = m21.converter.parse(os.path.join(path, file))
            tokens = tokenize_score(song)
            num_segments = (len(tokens) - segment_length) // step_size + 1
            for i in range(num_segments):
                segment = tokens[i * step_size : i * step_size + segment_length]
                if segment.count(-1) <= max_neg_ones:
                    songs.append(segment)
    return songs




def process_files(folder_path):
    element_dict = defaultdict(int)

    files = os.listdir(folder_path)
    files.sort()

    for file in files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'r') as f:
            content = f.read().split()

        for element in content:
            element_dict[element] += 1

    element_mapping = {element: i for i, element in enumerate(element_dict.keys())}
    reversed_mapping = {v: k for k, v in element_mapping.items()}  # Reverse the mapping

    return reversed_mapping

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

