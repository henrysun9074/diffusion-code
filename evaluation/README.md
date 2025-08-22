# CLIP Evaluation Script

This script evaluates synthetic images against a reference set of real species images using the CLIP (Contrastive Language-Image Pre-training) model.

## Overview

The script computes CLIP embeddings for real species images to create a reference centroid, then scores synthetic images against this centroid using cosine similarity. This allows you to assess how well synthetic images match the visual characteristics of the target species.


## Requirements

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python clip_evaluation.py \
    --species-folder data/real \
    --synthetic-folder data/synthetic \
    --similarity-threshold 0.3
```

### Arguments

- `--species-folder`: Path to folder containing real species images (default: `data/real`)
- `--synthetic-folder`: Path to folder containing synthetic images to evaluate (default: `data/synthetic`)
- `--similarity-threshold`: Cosine similarity threshold for pass/fail (default: 0.3)






