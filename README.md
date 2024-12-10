# GoogleSearchQueryRecommendationSystem


## Project Overview

This project focuses on developing and optimizing a machine learning pipeline to validate well-formed and malformed search queries in search recommendation systems. Using BERT embeddings and a Random Forest Classifier, the project aims to improve the detection of malformed queries, enhance model recommendations, and optimize overall system performance. Key techniques include leveraging pre-trained transformers, addressing class imbalances, and implementing robust feature engineering.

### Objectives and Goals

1. **Problem Definition**: Analyze and address the challenges associated with malformed queries in search recommendation systems.
2. **Optimization**: Utilize BERT embeddings and Leaky RELU activation to achieve high test accuracy for query validation.
3. **Impact**: Improve the robustness and efficiency of search query recommendation systems.

### Methodology

- **Dataset**: The `google_wellformed_query` dataset from Google Research was used, containing labeled queries indicating well-formed and malformed status.
- **Feature Engineering**: Extracted embeddings using BERT to generate meaningful numerical representations of text data.
- **Modeling**: Applied a Random Forest Classifier and regression analysis for effective query classification and validation.
- **Evaluation Metrics**: Metrics such as Mean Squared Error (MSE), R² Score, and accuracy were used to evaluate model performance.

### Results and Key Findings

- **Test Accuracy**: Achieved **78% accuracy** in validating queries as well-formed or malformed.
- **Validation Metrics**:
  - **Mean Squared Error (MSE)**: Captured the prediction error in numerical form.
  - **R² Score**: Indicated the proportion of variance explained by the model.
- **Model Robustness**: Addressed class imbalances to improve consistent detection of malformed queries.

### Potential Next Steps

1. Fine-tune the model with larger, more diverse datasets to enhance generalizability.
2. Implement a deployment pipeline to integrate the model into a live recommendation system.
3. Explore alternative transformer architectures (e.g., GPT, RoBERTa) for improved feature extraction.

---

## Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  
3. [Contributing](#contributing)  
4. [License](#license)  
5. [Credits and Acknowledgments](#credits-and-acknowledgments)  

---

## Installation

### Prerequisites

Ensure the following libraries and tools are installed in your environment:

- Python 3.8+
- [Datasets](https://huggingface.co/docs/datasets/)
- [Transformers](https://huggingface.co/docs/transformers/)
- Scikit-learn
- NumPy
- Pandas
- Torch

### Installation Commands

Run the following commands to set up your environment:

```bash
!pip install datasets transformers scikit-learn
```

---

## Usage

### Step-by-Step Instructions

1. **Load the Dataset**: Use the `google_wellformed_query` dataset from Google Research.
2. **Preprocess Data**: Tokenize content using BERT tokenizer and extract embeddings using a pre-trained BERT model.
3. **Train the Model**:
   - Split the data into training, validation, and test sets.
   - Train a Random Forest Classifier using content embeddings.
4. **Evaluate Performance**:
   - Calculate metrics such as MSE and R² Score on validation and test datasets.

### Code Snippet

```python
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from transformers import BertTokenizer, BertModel
import torch

# Load dataset
ds = load_dataset("google-research-datasets/google_wellformed_query")

# Tokenization and embedding extraction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Train a Random Forest Classifier
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
```

### Example Output

```
Validation Mean Squared Error: 0.15  
Validation R² Score: 0.82  
Test Mean Squared Error: 0.18  
Test R² Score: 0.79  
```

---

## Contributing

We welcome contributions to improve the project. To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Submit a pull request for review.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Credits and Acknowledgments

- **Team Members**: Krishi Shah, Brianna Anaya, Anju Soman, Nayera Hasan, Emily Yu
- **Advisors**: Belinda Shi, Chi Tong, Mako Ohara 
- **Tools**: Hugging Face Transformers, Scikit-learn, PyTorch.  

--- 

We hope this README serves as a detailed guide to understanding and utilizing our AI Studio Challenge project.
