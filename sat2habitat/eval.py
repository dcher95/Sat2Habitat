import open_clip
import torch
from torch.utils.data import DataLoader
from datasets import SatHabData
from config import config
import numpy as np
from tqdm import tqdm
import os
import random

import matplotlib.pyplot as plt
from PIL import Image
import pickle
from utils import get_county_polygon
from shapely.geometry import Point
import geopandas as gpd

from clip import CLIP
from crisp import CRISP
from crisp_exp import CRISPExp
from crisp_curr import CRISPCurr

def precompute_image_embeddings(model, data_loader):
    """
    Precompute normalized image embeddings for the image database.

    Args:
    - model (CRISP): The trained CRISP model.
    - data_loader (DataLoader): DataLoader for the image database.

    Returns:
    - image_embeddings (torch.Tensor): Normalized embeddings for all images.
    """
    model.eval()  # Set model to evaluation mode
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing embeddings", unit="batch"):
            # Forward pass to compute image embeddings
            combined_features, _, _ = model.forward(batch)
            
            # Collect normalized embeddings
            embeddings.append(combined_features)

    # Concatenate all embeddings into a single tensor
    return torch.cat(embeddings, dim=0)

def get_top_k_images(model, tokenizer, text, image_database, k=5):
    """
    Retrieve the top-k images given a text query using the CRISP model.
    
    Args:
    - model (CRISP): The trained CRISP model.
    - text (str): The input text query.
    - image_database (torch.Tensor): Precomputed normalized image embeddings, shape (N, embedding_dim).
    - k (int): The number of top images to retrieve.
    
    Returns:
    - top_k_indices (list): Indices of the top-k images.
    """
    model.eval()  # Set model to evaluation mode

    # Tokenize the input text
    text_tokens = tokenizer(text).squeeze(1)
    text_tokens = text_tokens.to(model.device)

    # Encode the text into embeddings
    with torch.no_grad():
        text_embedding = model.text_encoder.encode_text(text_tokens)
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)  # Normalize the embedding
    
    # Compute similarity scores
    similarities = text_embedding @ image_database.T  # Shape: (1, N)
    similarities = similarities.squeeze(0)  # Shape: (N,)

    # Ensure that k is not larger than the number of available embeddings
    k = min(k, similarities.size(0))

    # Retrieve top-k image indices
    top_k_indices = torch.topk(similarities, k=k).indices.cpu().tolist()
    return top_k_indices

def filter_embeddings(geoid_index, geoid_indices, precomputed_embeddings):
    """
    Optimized version of the filter_embeddings function using tensors for filtering.
    Filters embeddings based on matching 'geoid' values using tensor operations.

    Args:
    - row (pandas.Series): A row from the dataset containing the geoid.
    - geoid_indices (torch.Tensor): Tensor of numerical indices corresponding to geoids.
    - precomputed_embeddings (torch.Tensor): The precomputed image embeddings.

    Returns:
    - torch.Tensor: Filtered embeddings for the given geoid.
    """
    
    # Find matching indices for the geoid
    matching_indices = (geoid_indices == geoid_index).nonzero(as_tuple=True)[0]  # Return matching indices as a tensor

    # Filter the embeddings using the matching indices
    filtered_embeddings = precomputed_embeddings[matching_indices]
    
    # Return both filtered embeddings and the corresponding indices
    return filtered_embeddings, matching_indices

def visualize_images(image_paths, query_text):
    """
    Visualizes the top-k images and displays the query text.

    Args:
    - image_paths (list of str): Paths to the top-k images.
    - query_text (str): The query text used for retrieval.
    """
    plt.figure(figsize=(15, 10))
    
    # Display the query text
    print(f"Query: {query_text}")
    
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Top {i+1}")
    
    plt.show()

def evaluate_top_k_retrieval(trained_model, tokenizer, test_dataset, precomputed_embeddings, k_list=[1, 5, 10, 25], filtering=False):
    correct_counts = {k: 0 for k in k_list}  # Track counts of correct keys in top-k images
    total_queries = len(test_dataset)
    # total_queries = 500

    # Factorize the 'level2Gid' column and store the numerical indices in place
    geoid_list = test_dataset.data["level2Gid"].tolist()
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(set(geoid_list))}
    test_dataset.data["geoid_idx"] = test_dataset.data["level2Gid"].map(geoid_to_idx)

    # Convert the new 'geoid_idx' column to a tensor for efficient processing
    geoid_indices = torch.tensor(test_dataset.data["geoid_idx"].values, dtype=torch.long)

    top_k_results = {k: [] for k in k_list}  # To store the top-k indices for each query
    
    # Loop over all images in the test dataset
    for idx in tqdm(range(total_queries), desc="Evaluating top-k retrievals"):
        # Get the correct key for the current query
        row = test_dataset.data.iloc[idx]
        correct_key = row["key"].astype(int).astype(str)
        geoid_index = row["geoid_idx"].astype(int)

        # Query text generation
        query_text = row[config.hab_desc]  # Assuming this is the habitat description

        # Check if the shapefile path is provided. Then do filtering on the county boundary
        if filtering:
            # Retrieve top-k image indices for the current query text (for specific county)
            embeddings_filtered, filtered_indices = filter_embeddings(geoid_index, geoid_indices, precomputed_embeddings)
            top_image_indices = get_top_k_images(trained_model, tokenizer, query_text, embeddings_filtered, k=max(k_list))

            # Map the filtered top-k indices back to the image indices
            top_image_indices_original = filtered_indices[top_image_indices].tolist()
        else:
            # Retrieve top-k image indices for the current query text (against all images)
            top_image_indices = get_top_k_images(trained_model, tokenizer, query_text, precomputed_embeddings, k=max(k_list))
            top_image_indices_original = top_image_indices
        
        # Store top-k image indices
        for k in k_list:
            top_k_results[k].append(top_image_indices_original[:k])  # Store only the top-k results

        # Check if the correct key is in the top-k results for each k
        
        for k in k_list:
            if correct_key in [str(test_dataset.data.iloc[i]["key"]) for i in top_image_indices_original[:k]]:
                correct_counts[k] += 1
    
    # Calculate accuracy for each top-k list
    accuracy = {k: correct_counts[k] / total_queries for k in k_list} 
    
    # Print statistics
    print("Top-k Retrieval Accuracy:")
    for k in k_list:
        print(f"Top-{k} accuracy: {accuracy[k] * 100:.2f}%")

    return top_k_results, accuracy

def evaluate_retrieval_ecoregion_percentage(trained_model, tokenizer, test_dataset, precomputed_embeddings, k_list=[1, 5, 10, 25], region_col = "NA_L2CODE", filtering=False):
    
    # TODO: Predominately the correct eco-region? 
    # TODO: Is this being calculated correctly? It seems really low. 
    
    total_queries = len(test_dataset)

    # Factorize the 'level2Gid' column and store the numerical indices in place
    geoid_list = test_dataset.data["level2Gid"].tolist()
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(set(geoid_list))}
    test_dataset.data["geoid_idx"] = test_dataset.data["level2Gid"].map(geoid_to_idx)

    # Convert the new 'geoid_idx' column to a tensor for efficient processing
    geoid_indices = torch.tensor(test_dataset.data["geoid_idx"].values, dtype=torch.long)

    # Store the ecoregion code for each image
    ecoregion_codes = test_dataset.data[region_col].tolist()
    
    # Initialize storage for percentages
    ecoregion_percentages = {k: [] for k in k_list}  # Stores percentage of correct ecoregion matches for each query

    # Loop over all queries in the test dataset
    for idx in tqdm(range(total_queries), desc="Evaluating retrieval by ecoregion percentage"):
        # Get the correct ecoregion for the current query
        row = test_dataset.data.iloc[idx]
        correct_ecoregion = row[region_col]
        geoid_index = row["geoid_idx"] #TODO: Correct this?? Do we need filtering by county??

        # Query text generation
        query_text = row[config.hab_desc]  # Assuming this is the habitat description

        # Check if the shapefile path is provided. Then do filtering on the county boundary
        if filtering:
            # Retrieve top-k image indices for the current query text (for specific county)
            embeddings_filtered, filtered_indices = filter_embeddings(geoid_index, torch.tensor(ecoregion_codes), precomputed_embeddings)
            top_image_indices = get_top_k_images(trained_model, tokenizer, query_text, embeddings_filtered, k=max(k_list))
            # Map the filtered top-k indices back to the image indices
            top_image_indices_original = filtered_indices[top_image_indices].tolist()
        else:
            # Retrieve top-k image indices for the current query text (against all images)
            top_image_indices = get_top_k_images(trained_model, tokenizer, query_text, precomputed_embeddings, k=max(k_list))
            top_image_indices_original = top_image_indices

        # Calculate the percentage of top-k results in the correct ecoregion
        for k in k_list:
            retrieved_ecoregions = [
                test_dataset.data.iloc[i]["NA_L2CODE"]
                for i in top_image_indices_original[:k]
                if i < len(test_dataset.data)  # Ensure the index is valid
            ]
            correct_count = sum(ecoregion == correct_ecoregion for ecoregion in retrieved_ecoregions)
            percentage_correct = correct_count / k  # Percentage of top-k results in the correct ecoregion
            ecoregion_percentages[k].append(percentage_correct)

    # Aggregate metrics across all queries
    aggregated_percentages = {k: sum(ecoregion_percentages[k]) / total_queries for k in k_list}

    # Print statistics
    print("Top-k Retrieval Ecoregion Percentage:")
    for k in k_list:
        print(f"Top-{k} mean percentage: {aggregated_percentages[k] * 100:.2f}%")

    return ecoregion_percentages, aggregated_percentages

def evaluate_median_rank_recall(trained_model, tokenizer, test_dataset, precomputed_embeddings, filtering=False, sampling=1000):

    total_queries = len(test_dataset)
    sampled_indices = random.sample(range(total_queries), min(sampling, total_queries))  # Randomly sample 1000 queries or fewer if the dataset is smaller

    # Factorize the 'level2Gid' column and store the numerical indices in place
    geoid_list = test_dataset.data["level2Gid"].tolist()
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(set(geoid_list))}
    test_dataset.data["geoid_idx"] = test_dataset.data["level2Gid"].map(geoid_to_idx)

    # Convert the new 'geoid_idx' column to a tensor for efficient processing
    geoid_indices = torch.tensor(test_dataset.data["geoid_idx"].values, dtype=torch.long)

    ranks = []  # List to store the rank of the correct match for each query

    # Loop over the sampled queries
    for idx in tqdm(sampled_indices, desc="Evaluating median rank recall"):
        # Get the correct key for the current query
        row = test_dataset.data.iloc[idx]
        correct_key = row["key"].astype(int).astype(str)
        geoid_index = row["geoid_idx"].astype(int)

        # Query text generation
        query_text = row[config.hab_desc]  # Assuming this is the habitat description

        if filtering:
            # Retrieve filtered embeddings based on county boundary
            embeddings_filtered, filtered_indices = filter_embeddings(geoid_index, geoid_indices, precomputed_embeddings)
            top_image_indices = get_top_k_images(trained_model, tokenizer, query_text, embeddings_filtered, k=precomputed_embeddings.shape[0])

            # Map the filtered indices back to the original dataset indices
            ranked_image_indices = filtered_indices[top_image_indices].tolist()
        else:
            # Retrieve ranked indices for all images
            ranked_image_indices = get_top_k_images(trained_model, tokenizer, query_text, precomputed_embeddings, k=precomputed_embeddings.shape[0])
            # ranked_image_indices = get_top_k_images(trained_model, tokenizer, query_text, precomputed_embeddings, k=1000)

        # Determine the rank of the correct key
        for rank, image_idx in enumerate(ranked_image_indices, start=1):
            # Ensure image_idx is within bounds and corresponds to test_dataset
            if image_idx < len(test_dataset.data) and str(test_dataset.data.iloc[image_idx]["key"]) == correct_key:
                ranks.append(rank)
                break

    # Calculate the median rank
    median_rank = np.median(ranks) if ranks else float('inf')
    average_rank = np.average(ranks) if ranks else float('inf')

    # Print statistics
    print(f"Median rank: {median_rank:.2f}")
    print(f"Average rank: {average_rank:.2f}")

    return ranks, median_rank, average_rank


# Example usage
if __name__ == "__main__":

    embedding_file_path = config.embedding_file_path

    if 'clip' in config.experiment_model_path:
        # Baseline
        trained_model = CLIP.load_from_checkpoint(config.experiment_model_path,
                                                train_dataset=None,
                                                val_dataset=None)
    elif 'exp' in config.experiment_model_path:
        trained_model = CRISPExp.load_from_checkpoint(config.experiment_model_path,
                                                    train_dataset=None,
                                                    val_dataset=None)
    elif '250' in config.experiment_model_path:
        # Baseline CRISP
        trained_model = CRISP.load_from_checkpoint(config.experiment_model_path,
                                                   train_dataset=None,
                                                   val_dataset=None)
    elif 'curr' in config.experiment_model_path:
        trained_model = CRISPCurr.load_from_checkpoint(config.experiment_model_path,
                                                    train_dataset=None,
                                                    val_dataset=None)
    else:
        trained_model = CRISPExp.load_from_checkpoint(config.experiment_model_path,
                                                    train_dataset=None,
                                                    val_dataset=None)
        
    trained_model.to('cuda') 

    test_dataset = SatHabData(config.im_dir_test, 
                              config.test_csv_path, 
                              mode='test')


    ### Precompute embeddings (if necessary) ###
    # Check if the precomputed embeddings already exist
    if not os.path.exists(embedding_file_path):
        
        # If the file doesn't exist, precompute the embeddings
        data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        precomputed_embeddings = precompute_image_embeddings(trained_model, data_loader)
        
        # Save the precomputed embeddings to the specified file
        torch.save(precomputed_embeddings, embedding_file_path)
        print(f"Precomputed embeddings saved to {embedding_file_path}")
    else:
        precomputed_embeddings = torch.load(embedding_file_path).to('cuda')
        print(f"Precomputed embeddings already exist at {embedding_file_path}. Skipping computation.")

    ### Evaluate Top-K Retrieval on all US ###
    tokenizer = open_clip.get_tokenizer('hf-hub:MVRL/taxabind-vit-b-16')
    # top_k_results, accuracy = evaluate_top_k_retrieval(trained_model, 
    #                                                    tokenizer, 
    #                                                    test_dataset, 
    #                                                    precomputed_embeddings,
    #                                                    filtering = False)
    

    # ### Evaluate Top-K Retrieval by county ###
    # top_k_results_county, accuracy_county = evaluate_top_k_retrieval(trained_model, 
    #                                                    tokenizer, 
    #                                                    test_dataset, 
    #                                                    precomputed_embeddings,
    #                                                    filtering = True)  

    # # Save the results to a pickle file
    # with open(f"{config.metric_save_path}.pkl", 'wb') as f:
    #     pickle.dump({'top_k_results': top_k_results, 'accuracy': accuracy}, f)
    #     print(f"Top-k results, accuracy saved to {config.metric_save_path}.pkl")

    # with open(f"{config.metric_save_path}_county.pkl", 'wb') as f:
    #     pickle.dump({'top_k_results': top_k_results_county, 'accuracy': accuracy_county}, f)
    #     print(f"Top-k results, accuracy saved to {config.metric_save_path}_county.pkl")

    # Evaluate Median Rank of Recall
    ranks, median_rank, average_rank = evaluate_median_rank_recall(trained_model, 
                                                       tokenizer, 
                                                       test_dataset, 
                                                       precomputed_embeddings,
                                                       filtering = False)
    

    with open(f"{config.metric_save_path}_rank.pkl", 'wb') as f:
        pickle.dump({'ranks': ranks, 'median_rank': median_rank, "average_rank" : average_rank}, f)
        print(f"Median and Average rank saved to {config.metric_save_path}_rank.pkl")

    ### Evaluate Top-K Retrieval by ecoregion ###
    # top_k_results_eco, accuracy_eco = evaluate_retrieval_ecoregion_percentage(trained_model, 
    #                                                    tokenizer, 
    #                                                    test_dataset, 
    #                                                    precomputed_embeddings,
    #                                                    filtering = False)
    
    # with open(f"{config.metric_save_path}_eco.pkl", 'wb') as f:
    #     pickle.dump({'top_k_results': top_k_results_eco, 'accuracy': accuracy_eco}, f)
    #     print(f"Top-k results and accuracy saved to {config.metric_save_path}_eco.pkl")
    
    # ### Evaluate Top-K Retrieval by ecoregion county ###
    # top_k_results_county_eco, accuracy_county_eco = evaluate_retrieval_ecoregion_percentage(trained_model, 
    #                                                    tokenizer, 
    #                                                    test_dataset, 
    #                                                    precomputed_embeddings,
    #                                                    filtering = True)

    # with open(f"{config.metric_save_path}_county_eco.pt", 'wb') as f:
    #     pickle.dump({'top_k_results': top_k_results_county_eco, 'accuracy': accuracy_county_eco}, f)
    #     print(f"Top-k results and accuracy saved to {config.metric_save_path}_county_eco.pkl")


    # How to show that it builds a more granular eco-region map?
    # 1. For each row, get the eco-region
    # 2. Run inference for each row
    # 3. Get the top-k results
    # 4. See if the top-k results are in the same eco-region
    # 5. Calculate the accuracy for each eco-region
        # Per-eco-region accuracy to identify eco-regions where the model excels or struggles.
        # Overall precision, recall, and F1 score across eco-regions.
        # Eco-region diversity index: Are certain eco-regions consistently overrepresented or underrepresented in the retrieval results?

    # T-SNE visualization of the embeddings of different eco-regions??




    ############################################################
    # ### Sample:: Retrieve top-k images given a query text ###
    
    # # Query text
    # query_text = "A forest with dense trees and sunlight streaming through"

    # # Retrieve top 5 images
    # top_image_indices = get_top_k_images(trained_model, tokenizer, query_text, precomputed_embeddings, k=5)
    # print(f"Top 5 image indices: {top_image_indices}")

    # ### Visualization of Top-K Images ###
    
    # top_image_paths = []
    # for idx in top_image_indices:
    #     row = test_dataset.data.iloc[idx]
    #     sat_id = row["key"].astype(int).astype(str)
    #     image_path = test_dataset.image_dict.get(sat_id)  # Get the image file path using the sat_id
    #     print(image_path)
    #     top_image_paths.append(image_path)
    
    # visualize_images(top_image_paths, query_text)
