import music21 as m21
import os
import random
import torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch.backends.cudnn.enabled = False
from lstm_module import LSTMModule
from data_processing import tokenize_score, process_files, one_hot_encoding, load_tokenized_songs


def generate_melody(model, input_midi_path, output_length=100, temperature=1.0):
    # Load the MIDI file and tokenize it
    input_tokens = tokenize_score(m21.converter.parse(input_midi_path))
    # Convert the tokens to a tensor and add an extra dimension for batch size
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)

    # Send the input tensor to the same device as the model
    input_tensor = input_tensor.to(model.device)

    # Initialize an empty list to store the output tokens
    output_tokens = []

    # Feed the input to the model and generate the output tokens
    with torch.no_grad():
        for _ in range(output_length):

            # Forward pass through the model
            output = model(input_tensor)
            #print("output: ", output.shape)
            # Apply temperature to the output distribution and sample a token
            output = output / temperature
            probabilities = torch.nn.functional.softmax(output, dim=1)
            token = torch.multinomial(probabilities, num_samples=1)

            # Append the token to the output list and update the input tensor
            output_tokens.append(token.item())
            input_tensor = torch.cat([input_tensor, token], dim=1)

    # Convert the output tokens back to MIDI notes
    output_score = m21.stream.Stream()
    for token in output_tokens:
        if token == -1:
            output_score.append(m21.note.Rest(quarterLength=0.25))
        else:
            output_score.append(m21.note.Note(midi=token, quarterLength=0.25))

    return output_score

random.seed(1)
    # Load and process the dataset
dataset_path = "/data/csq/adl-piano-midi/small_classical"  # Update the dataset path accordingly
desired_length = 100
songs = load_tokenized_songs(dataset_path)
print(f"Loaded {len(songs)} songs.")
folder_path = "/data/csq/converted_songs_small_1"  # Update the save folder path accordingly
os.makedirs(folder_path, exist_ok=True)
for i, song in enumerate(songs):
    converted_song = " ".join(map(str, song))
    save_path = os.path.join(folder_path,  str(i))
    with open(save_path, "w") as fp:
        fp.write(converted_song)
# Process the dataset and perform one-hot encoding
element_mapping = process_files(folder_path)
print(len(element_mapping))
encoded_arrays = one_hot_encoding(folder_path, element_mapping, desired_length)
# Initialize the LSTM module
# input_size = len(element_mapping)
input_size = 76
hidden_size = 128
# output_size = len(element_mapping)
output_size = 76
dropout_prob = 0.2
model = LSTMModule(input_size, hidden_size, output_size, dropout_prob)
print(model)
# Set hyperparameters and train the model
batch_size = 16
num_epochs = 50
learning_rate = 0.0001
#train_model(model, element_mapping, encoded_arrays, batch_size, num_epochs, learning_rate)
model.load_state_dict(torch.load('classical_0730.pth'))  # Load the model weights from a file
model.eval()  # Set the model to evaluation mode
output_score = generate_melody(model, '/data/csq/adl-piano-midi/classical_1.mid')
output_score.write('midi', fp='output.mid')  # Save the output as a MIDI file
