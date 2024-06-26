# Laptop Price Prediction

Predicting laptop prices based on various features using machine learning regression models.

## Overview

This project aims to predict the prices of laptops using machine learning regression techniques. It involves preprocessing the dataset, training multiple regression models, evaluating their performance, and predicting prices for new laptop specifications entered interactively by the user.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Models Used](#models-used)
- [License](#license)
- [Author](#author)

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/laptop-price-prediction.git
   cd laptop-price-prediction
Install dependencies:

Make sure you have Python 3.x installed. Install required libraries using pip:

bash
Copy code
pip install -r requirements.txt
This will install the necessary libraries such as pandas, scikit-learn, and their dependencies.

Dataset

Place your dataset file (laptop_price0.csv) in the project directory.

Usage
Run the script main.py to train regression models, perform evaluations, and predict laptop prices for new input data.

bash
Copy code
python main.py
Follow the prompts to enter details for a new laptop and observe the predicted prices from different regression models including Random Forest, Linear Regression, Decision Tree, Extra Trees, and Ridge Regression.

File Structure
main.py: Main script for loading data, preprocessing it, training regression models, evaluating performance, and predicting prices for new data.
laptop_price0.csv: Dataset file containing laptop specifications and prices.
README.md: This file, providing project overview, setup instructions, usage guidelines, and file structure.
requirements.txt: Lists Python libraries required for the project.
Models Used
Random Forest
Linear Regression
Decision Tree
Extra Trees
Ridge Regression
License
This project is licensed under the MIT License. See the LICENSE file for details
