# CoFARS Implementation for Ecommerce Recommendation

Context-based Fast Recommendation Strategy for Long User Behavior Sequences

## Project Structure

```
.
├── data/                          # Raw data (DO NOT MODIFY)
│   ├── home_life_products_details.csv
│   └── home_life_reviews.csv
├── processed_data/               # Processed data (generated)
│   ├── cleaned/                  # Cleaned data
│   ├── contexts/                 # Context-related data
│   ├── sequences/                # User sequences
│   └── splits/                   # Train/val/test splits
├── results/                      # Results for paper
│   ├── figures/                  # Visualizations
│   ├── statistics/               # Statistical analyses
│   └── models/                   # Trained model checkpoints
├── src/                          # Source code
│   ├── data_processing/          # Data preprocessing modules
│   ├── models/                   # Model implementations
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
├── logs/                         # Training/processing logs
└── notebooks/                    # Jupyter notebooks for analysis

## Implementation Details

**Context Strategy**: Option B (time_slot + is_weekend)
- Number of contexts: 10
- Examples: morning_weekday, afternoon_weekend, evening_weekday, etc.

**Product Attributes for JS Divergence**:
- Category (group field)
- Price bucket (price_bucket field)
- Rating (bucketed into Low/Medium/High)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preprocessing
```bash
python src/data_processing/01_clean_data.py
python src/data_processing/02_build_contexts.py
python src/data_processing/03_calculate_divergence.py
python src/data_processing/04_create_sequences.py
```

### Step 2: Training
```bash
python src/train.py --config configs/model_config.yaml
```

### Step 3: Evaluation
```bash
python src/evaluate.py --checkpoint results/models/best_model.pt
```

## Results

Results are saved in `results/` directory for paper writing:
- Statistics: `results/statistics/`
- Figures: `results/figures/`
- Model checkpoints: `results/models/`
