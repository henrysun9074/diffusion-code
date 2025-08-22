#!/usr/bin/env python3
import os
import argparse
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Set up device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model 
model, preprocess = clip.load("ViT-B/32", device=device)

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, file)
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            images.append(image)
            filenames.append(file)
        except Exception as e:
            print(f"Skipping {file}: {e}")
    return images, filenames

def compute_clip_embeddings(images):
    with torch.no_grad():
        image_batch = torch.cat(images, dim=0)
        image_features = model.encode_image(image_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def compute_species_centroid(species_folder):
    images, _ = load_images_from_folder(species_folder)
    if not images:
        raise ValueError(f"No valid images found in {species_folder}")
    embeddings = compute_clip_embeddings(images)
    centroid = embeddings.mean(dim=0, keepdim=True)
    return centroid

def score_synthetic_images_against_species(synthetic_dir, species_centroid, similarity_threshold=0.3):
    images, filenames = load_images_from_folder(synthetic_dir)
    results = []

    if not images:
        return pd.DataFrame(results)

    synth_embeddings = compute_clip_embeddings(images)

    for i, emb in enumerate(synth_embeddings):
        similarity = cosine_similarity(emb.unsqueeze(0).cpu(), species_centroid.cpu())[0][0]
        passed = similarity >= similarity_threshold
        results.append({
            "filename": filenames[i],
            "similarity_score": similarity,
            "passed_threshold": passed
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Score synthetic images against a species centroid using CLIP.")
    parser.add_argument(
        "--species-folder",
        dest="species_folder",
        default="data/real",
        help="Folder with real species images"
    )
    parser.add_argument(
        "--synthetic-folder",
        dest="synthetic_folder",
        default="data/synthetic",
        help="Folder with synthetic images to score"
    )
    parser.add_argument(
        "--similarity-threshold",
        dest="similarity_threshold",
        type=float,
        default=0.3,
        help="Cosine similarity threshold (default: 0.3)."
    )
    args = parser.parse_args()

    # Compute the reference centroid 
    centroid = compute_species_centroid(args.species_folder)

    # Score synthetic images 
    results_df = score_synthetic_images_against_species(
        args.synthetic_folder, centroid, similarity_threshold=args.similarity_threshold
    )

    print(results_df)

if __name__ == "__main__":
    main()


# Usage:
# python clip_evaluation.py --species-folder data/real --synthetic-folder data/synthetic --similarity-threshold 0.3