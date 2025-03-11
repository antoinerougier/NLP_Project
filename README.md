# Movie Sentiment Analysis Project

## Overview

This project focuses on sentiment analysis of movie reviews. We leverage techniques from natural language processing (NLP) to analyze and predict sentiments expressed in movie reviews. The project is based on the methodologies outlined in the paper "Learning Word Vectors for Sentiment Analysis" by Andrew L. Maas et al.

## Setup Instructions

### Installation

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/antoinerougier/NLP_Project.git>
   cd <NLP_Project>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Data

The dataset is provided in a `.tar` format. The preprocessing step involves downloading and extracting this data.

### Running the Project

To run the project, simply execute the `main.py` script. This script will handle the entire workflow, including:

- Downloading the dataset
- Extracting and formatting the data
- Training the sentiment analysis model
- Evaluating the model performance

```bash
python main.py
```

### Project Structure

- **`src/preprocessing/download_data.py`**: Script for downloading and extracting the dataset.
- **`src/preprocessing/pre_processing.py`**: Script for additional data preprocessing.
- **`src/preprocessing/dataframe_creation.py`**: Script for creating a DataFrame from the processed data.
- **`src/model.py`**: Script containing the model training and evaluation logic.
- **`main.py`**: The main script to run the entire pipeline.

### Notebook

The `Notebook` directory contains Jupyter notebooks used for testing and exploring different approaches. These notebooks are not essential for running the main project but can be useful for understanding the development process.

## Acknowledgements

This project is inspired by the paper "Learning Word Vectors for Sentiment Analysis" by Andrew L. Maas et al. We appreciate their contributions to the field of NLP and sentiment analysis.
