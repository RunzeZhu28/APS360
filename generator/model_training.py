import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import random
def train_model(model, element_mapping, encoded_arrays, batch_size, num_epochs, learning_rate):
    # Split dataset into training and validation sets
    random.shuffle(encoded_arrays)
    split_ratio = 0.7
    split_index = int(split_ratio * len(encoded_arrays))
    train_data = encoded_arrays[:split_index]
    val_data = encoded_arrays[split_index:]

    # Convert one-hot encoded arrays to PyTorch tensors
    train_tensors = [(torch.tensor(array), file) for array, file in train_data]
    val_tensors = [(torch.tensor(array), file) for array, file in val_data]

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store loss values
    train_loss = []
    val_loss = []

    # Create loss file
    with open("electronic_loss_values.txt", "w") as f:
        f.write("Epoch\tTrain Loss\tValidation Loss\n")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Iterate over training data in batches
        for i in range(0, len(train_tensors), batch_size):
            batch_data = train_tensors[i:i + batch_size]
            inputs, labels = zip(*batch_data)
            inputs = torch.stack(inputs).cuda()
            labels = torch.tensor([element_mapping.get(file, 0) for file in labels]).cuda()

            # Zero gradients, perform forward pass, calculate loss, and perform backpropagation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / (len(train_tensors) / batch_size)
        train_loss.append(average_loss)

        # Validate the model on the validation set
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(val_tensors), batch_size):
                batch_data = val_tensors[i:i + batch_size]
                inputs, labels = zip(*batch_data)
                inputs = torch.stack(inputs).cuda()
                labels = torch.tensor([element_mapping.get(file, 0) for file in labels]).cuda()

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / (len(val_tensors) / batch_size)
        val_loss.append(average_val_loss)

        # Print training progress
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f} - Val Loss: {average_val_loss:.4f}")

        # Save loss values to file
        with open("electronic_loss_values.txt", "a") as f:
            f.write(f"{epoch + 1}\t{average_loss}\t{average_val_loss}\n")

    # Plotting the training curve
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save model parameters
    torch.save(model.state_dict(), 'electronic.pth')
    #model = LSTMModule(input_size, hidden_size, output_size, dropout_prob)  # The model must be defined first
    #model.load_state_dict(torch.load('model_parameters.pth'))

