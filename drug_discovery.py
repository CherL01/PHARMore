import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def train_autoencoder(ratings_file="./ratings_mat.csv"):
    # Load datasets
    # ratings_file = "/content/drive/My Drive/Hacklytics/Datasets/ratings_mat.csv"
    # mappings_file = "/content/drive/My Drive/Hacklytics/Datasets/drug-mappings.tsv"

    ratings_df = pd.read_csv(ratings_file, index_col=0)  # Drugs as rows & cols
    # mappings_df = pd.read_csv(mappings_file, delimiter="\t")  # Drug mappings

    # Ensure drug IDs are consistent (convert to string to avoid float IDs)
    ratings_df.index = ratings_df.index.astype(str)
    ratings_df.columns = ratings_df.columns.astype(str)

    # Normalize data (scale to [0,1] range)
    scaler = MinMaxScaler()
    ratings_matrix = scaler.fit_transform(ratings_df.values)

    # Define Autoencoder Model
    class DrugAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=64):
            super(DrugAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim)  # Bottleneck layer
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    # Convert to PyTorch tensor
    ratings_tensor = torch.tensor(ratings_matrix, dtype=torch.float32)

    # Model & Training Setup
    input_dim = ratings_matrix.shape[1]
    autoencoder = DrugAutoencoder(input_dim, latent_dim=64)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    # Train Autoencoder
    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        encoded, decoded = autoencoder(ratings_tensor)
        loss = loss_function(decoded, ratings_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Extract Latent Representations
    with torch.no_grad():
        latent_embeddings, _ = autoencoder(ratings_tensor)

    # Compute Cosine Similarities in Latent Space
    latent_matrix = latent_embeddings.numpy()
    cosine_sim_matrix = cosine_similarity(latent_matrix)

    # Convert similarity matrix to DataFrame
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=ratings_df.index, columns=ratings_df.index)

    return cosine_sim_df

# Function to get most similar drugs
def get_similar_drugs_autoencoder(drug_id, sim_df, top_n=5):
    if drug_id not in sim_df.index:
        return f"Drug ID {drug_id} not found in dataset"

    # similar_drugs = sim_df[drug_id].sort_values(ascending=False).iloc[1:top_n+1]
    similar_drugs = sim_df[drug_id].sort_values(ascending=False).iloc[1:top_n+1].reset_index()
    similar_drugs.columns = ["Drug", "Similarity Score"]
    return similar_drugs

# # Example Usage
# sim_df = train_autoencoder()
# drug_id_example = "54675785"  # Example DrugBank ID or PubChem CID
# print(f"Most similar drugs to {drug_id_example}:")
# print(get_similar_drugs_autoencoder(drug_id_example, sim_df))
