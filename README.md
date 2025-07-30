# Fake News Detection System

A high-performance fake news classifier built with Python and Scikit-Learn. This project uses Natural Language Processing (NLP) and a comparative analysis of machine learning models to achieve over 99% accuracy in detecting misinformation.

## Overview

This project provides an end-to-end solution for building and evaluating a high-performance fake news classifier. Utilizing a dataset of over 44,000 articles, the system employs a rigorous data science workflow, from data preparation and feature engineering to model training and comparative analysis. The final system dynamically selects the best-performing model to achieve an accuracy of **99.76%**, making it a robust tool in the fight against misinformation.

## Technologies Used

* **Python 3.9+**
* **Jupyter Notebook**
* **Core Libraries:**
    * **Scikit-Learn:** For machine learning models and metrics.
    * **Pandas:** For data manipulation and analysis.
    * **NLTK:** For Natural Language Processing tasks.
    * **Matplotlib & Seaborn:** For data visualization.
    * **Tqdm:** For creating progress bars.

## Features

**High-Accuracy Classification:** Achieves over 99% accuracy in distinguishing between real and fake news.

**Comparative Model Analysis:** Implements and evaluates a baseline `Logistic Regression` model against a powerful `Random Forest Classifier` to ensure the best model is chosen.

**Dynamic Model Selection:** Automatically identifies and uses the superior model for live predictions based on performance metrics.

**Advanced NLP Pipeline:** Includes a full text preprocessing workflow with normalization, stopword removal, and stemming.

**TF-IDF Feature Engineering:** Converts text into meaningful numerical features to feed the machine learning models.

**Interactive User Experience:** Incorporates a `tqdm` progress bar to provide real-time feedback during computationally intensive tasks.

**Rich Visualizations:** Generates clear and insightful plots for data distribution, model performance (Confusion Matrices), and comparative analysis.

**Data-Driven Reporting:** Dynamically generates easy-to-read tables and written analyses of model performance.

## Methodology & Pipeline

The project follows a structured 11-step data science workflow:

**Setup & Installation:** Preparing the environment by installing all necessary Python libraries.

**Library Imports:** Importing libraries and initializing components like `tqdm` for pandas.

**Data Loading & Preparation:** Loading the `true.csv` and `fake.csv` datasets and merging them into a single, shuffled DataFrame.

**Exploratory Data Analysis (EDA):** Visualizing the class distribution to ensure the dataset is balanced.

**Text Preprocessing:** Cleaning and standardizing the raw article text.

**Feature Engineering:** Applying TF-IDF vectorization to create a numerical feature matrix.

**Model A (Baseline):** Training and evaluating the `Logistic Regression` model.

**Model B (Advanced):** Training and evaluating the `Random Forest Classifier`.

**Model Comparison:** Visually and quantitatively comparing the performance of both models.

**Live Prediction:** Implementing a function to classify new, unseen articles using the dynamically selected best model.

**Final Conclusion:** Programmatically generating a summary of the project's findings.

## Performance & Results

The comparative analysis clearly demonstrated the superiority of the Random Forest Classifier.

| **Classifier** | **Precision** | **Recall** | **F1-Score** | **Accuracy** |
| --------------------- | ------------- | ---------- | ------------ | ------------ |
| **Logistic Regression** | 0.9878        | 0.9878     | 0.9878       | 0.9878       |
| **Random Forest** | **0.9976** | **0.9976** | **0.9976** | **0.9976** |


## Output

### Model Performance Comparison
*(This chart visually compares the key metrics of the Logistic Regression and Random Forest models, highlighting the superior performance of the Random Forest.)*
<img width="1157" height="622" alt="Untitled" src="https://github.com/user-attachments/assets/32e11a67-8945-46a9-b6b6-f8f787180c74" />

### Confusion Matrix (Random Forest)
*(This matrix shows the high accuracy of the final model, with very few misclassifications on the test data.)*
<img width="644" height="544" alt="Untitled-1" src="https://github.com/user-attachments/assets/c57b083b-3f93-46af-b2fd-e0e25a27a490" />

## Conclusion

This project successfully demonstrates a robust, data-driven approach to building a high-accuracy Fake News Detection System. By comparing a baseline `Logistic Regression` model with an advanced `Random Forest Classifier`, we confirmed that the ensemble method provides superior performance, achieving **99.76%** accuracy. The system is a powerful example of how NLP and machine learning can be effectively applied to combat misinformation.

## Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/Fake-News-Detection-System.git
    cd Fake-News-Detection-System
    ```

2.  **Place Datasets:**
    Ensure the `true.csv` and `fake.csv` files are located in the root directory of the cloned project.

3.  **Launch Jupyter Notebook:**
    Navigate to the project directory in your terminal and run:
    ```sh
    jupyter notebook
    ```

4.  **Open the Notebook:**
    In the Jupyter interface that opens in your browser, click on `Fake_News_Detection_System.ipynb` to open it.

5.  **Run the Cells:**
    -   Execute the cells in order from top to bottom.
    -   The first code cell will automatically install all the necessary Python packages using `pip`.
    -   **Important:** After the first cell finishes, **restart the kernel** from the menu (`Kernel` > `Restart`) before running the rest of the notebook. This ensures all newly installed libraries are correctly loaded.

## File Structure

The project is organized into a few key files:

-   `fake_news_detection_system.ipynb`: The main Jupyter Notebook containing the entire data science workflow, from data loading to the final conclusion.
-   `fake_news_detection_system.zip`:
      - `true.csv`: The dataset containing real news articles.
      - `fake.csv`: The dataset containing fake news articles.
-   `README.md`: This file.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

