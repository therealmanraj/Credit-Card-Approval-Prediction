# Credit Card Approval Prediction

![banner](assets/Credit_card_approval_banner.png)
Banner [source](https://banner.godori.dev/)

## Key Findings

People with the highest income, and who have at least one partner, are more likely to be approved for a credit card.

## Authors

- [Manraj Singh](https://www.github.com/therealmanraj)

## Table of Contents

- [Key Findings](#key-findings)
- [Authors](#authors)
- [Table of Contents](#table-of-contents)
- [Business Problem](#business-problem)
- [Data Source](#data-source)
- [Methods](#methods)
- [Tech Stack](#tech-stack)
- [Metrics Used](#metrics-used-recall)
- [Run Locally](#run-locally-windows--macoslinux)
- [Explore the Notebook](#explore-the-notebook)
- [Repository Structure](#repository-structure)
- [Contribution](#contribution)
- [License](#license)

## Business Problem

This app predicts if an applicant will be approved for a credit card or not. Each time there is a hard enquiry, your credit score is affected negatively. This app predicts the probability of being approved without affecting your credit score. This app can be used by applicants who want to find out if they will be approved for a credit card without affecting their credit score.

## Data Source

- [Kaggle Credit Card Approval Prediction](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)

## Methods

- Exploratory Data Analysis
- Bivariate Analysis
- Multivariate Correlation
- Model Deployment

## Tech Stack

- Python (refer to requirements.txt for the packages used in this project)

## Metrics Used: Recall

### Why Choose Recall as a Metric

Since the objective of this problem is to minimize the risk of credit default for financial institutions, the metric to use depends on the current economic situation:

- **Bull Market**: In a thriving economy, financial institutions may prioritize recall (sensitivity) to approve more applicants.
- **Bear Market**: During economic downturns, financial institutions may prioritize precision (specificity) to reduce the risk of defaults.

### Conclusion

Given the current economic conditions, we prioritize recall as our metric.

## Run Locally (Windows & macOS/Linux)

### 1. Initialize Git

```bash
git init
```

### 2. Clone the Project

```bash
git clone https://github.com/therealmanraj/Credit-Card-Approval-Prediction
```

### 3. Enter the Project Directory

```bash
cd Credit-Card-Approval-Prediction
```

### 4. Create a Virtual Environment and Install Dependencies

#### Using Conda (Recommended)

- **macOS/Linux**:

```bash
conda env create --prefix ./env --file assets/environment.yml
```

- **Windows**:

```bash
conda env create --prefix .\env --file assets\environment.yml
```

#### Using venv (If Conda Is Not Available)

- **macOS/Linux**:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

- **Windows**:

```bash
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

### 5. Activate the Virtual Environment

#### Using Conda

- **macOS/Linux**:

```bash
conda activate ./env
```

- **Windows**:

```bash
conda activate .\env
```

#### Using venv

- **macOS/Linux**:

```bash
source env/bin/activate
```

- **Windows**:

```bash
.\env\Scripts\activate
```

### 6. List Installed Packages

```bash
conda list
```

OR, if using venv:

```bash
pip list
```

## Explore the Notebook

Explore the notebook file [here](https://nbviewer.org/github/therealmanraj/Credit-Card-Approval-Prediction/blob/main/notebooks/CCAP.ipynb)

## Repository Structure

```
Credit-Card-Approval-Prediction
├── assets
├── notebooks
├── data
└── README.md
```

## Contribution

Contributions are welcome! Please open a pull request or issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
