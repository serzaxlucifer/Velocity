import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# # # # # # # # # # # # # # # # # # # # # # # # #
# Module: MULTI-VIEW NEURAL NETWORK (MVNN)      #
# Authors: Mukul Malik, Mukul Verma             #
# # # # # # # # # # # # # # # # # # # # # # # # #

class MultiviewNetwork(nn.Module):
    def __init__(self, embedding_dim, FCNeurons_current, CONVNeurons_current, FCNeurons_history, CONVNeurons_history, num_final_hidden, num_ratings, lr, eps, device=None):
        super(MultiviewNetwork, self).__init__()  
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.num_ratings = num_ratings
        self.epochs = eps

        # Model initialization
        total_neurons = ((2 * FCNeurons_current) + (2 * FCNeurons_history) + (10 * CONVNeurons_current) + (10 * CONVNeurons_history))

        self.num_final_hidden = num_final_hidden

        # Create sub-networks
        self.subnetworks = nn.ModuleList([
            self.create_subnetwork(embedding_dim, FCNeurons_current) if i < 2 else
            self.create_subnetwork(embedding_dim, CONVNeurons_current) if i < 12 else
            self.create_subnetwork(embedding_dim, FCNeurons_history) if i < 14 else
            self.create_subnetwork(embedding_dim, CONVNeurons_history) for i in range(24)
        ])

        # Fully connected layers
        self.fc1 = nn.Linear(total_neurons, num_final_hidden)
        self.fc2 = nn.Linear(num_final_hidden, 1)

        # Move the model to the appropriate device
        self.to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

        print("Model initialized and moved to", self.device)

    # Sub-network factory
    @staticmethod
    def create_subnetwork(input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, inputs):
        sub_outputs = [subnet(inputs[:, i, :]) for i, subnet in enumerate(self.subnetworks)]
        concatenated = torch.cat(sub_outputs, dim=1)  # Concatenate outputs
        hidden = F.relu(self.fc1(concatenated))
        output = self.fc2(hidden)
        return output

    def prepare_data(self, inputs, outputs):
        # Generate random data for demonstration (replace with your actual dataset)
        self.inputs = inputs.to(self.device)  
        self.targets = outputs.to(self.device)  

        # Create DataLoader for batching
        dataset = TensorDataset(inputs, outputs)
        return DataLoader(dataset, batch_size=64, shuffle=True)

    def train_model(self, inputs, outputs, num_epochs=10):
        train_loader = self.prepare_data(inputs, outputs)
        best_loss = float('inf')
        best_model_weights = None

        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0

            # Training phase
            for input_batch, target_batch in train_loader:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output = self(input_batch)

                # Loss calculation
                loss = self.criterion(output, target_batch)
                loss.backward()

                # Update weights
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase (reuse train_loader for simplicity here)
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for input_batch, target_batch in train_loader:
                    input_batch = input_batch.to(self.device)
                    target_batch = target_batch.to(self.device)

                    output = self(input_batch)
                    loss = self.criterion(output, target_batch)
                    val_loss += loss.item()

            val_loss /= len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the model if this epoch has the best validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = self.state_dict()

        # Load the best weights
        self.load_state_dict(best_model_weights)
        print("Training completed. Best model loaded.")
        
    def predict(self, inputs):
        """
        Predict the output for a given input.

        :param inputs: A tensor of shape (batch_size, 24, embedding_dim).
        :return: Model predictions as a tensor.
        """
        self.eval()  # Set model to evaluation mode
        inputs = inputs.to(self.device)  # Ensure inputs are on the correct device
        with torch.no_grad():
            outputs = self(inputs)  # Perform forward pass
        return outputs.cpu()  # Return results on CPU for easier handling (IMPORTANT! We spent quite some time here :))