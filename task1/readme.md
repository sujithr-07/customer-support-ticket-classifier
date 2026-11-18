
## Dataset Description

### Project Overview

This project uses a support ticket dataset to build a machine learning model that predicts:

1. **Category:** The type of customer support issue.
2. **Priority:** The urgency level of the support ticket.

### Dataset Source

The dataset is synthetically generated for this project. It contains 500 support ticket records and does not include real customer information.

### Dataset Features

| Feature  | Description                         | Type        |
| -------- | ----------------------------------- | ----------- |
| text     | Customer support ticket description | Text        |
| category | Support ticket category             | Categorical |
| priority | Ticket priority level               | Categorical |

### Dataset Statistics

* Total records: 500
* Unique records: 500
* Number of categories: 5
* Number of priority classes: 4
* Duplicate records removed: 0

### Category Distribution

| Category          | Count |
| ----------------- | ----- |
| Technical Support | 100   |
| Billing           | 100   |
| Account Access    | 100   |
| Shipping          | 100   |
| Product Inquiry   | 100   |

### Priority Distribution

| Priority | Count |
| -------- | ----- |
| Low      | 125   |
| Medium   | 125   |
| High     | 125   |
| Critical | 125   |

### Data Preprocessing

The following preprocessing steps were performed:

* Converted ticket text to lowercase.
* Removed leading and trailing whitespace.
* Removed duplicate ticket descriptions.
* Checked category and priority class distributions.
* Used stratified sampling based on the combination of category and priority.

### Dataset Splitting

The dataset was divided into three subsets:

| Dataset         | Percentage     | Records       |
| --------------- | -------------- | ------------- |
| Training        | 70%            | 350           |
| Validation      | 15%            | 75            |
| Testing         | 15%            | 75            |
| **Total** | **100%** | **500** |

A fixed random seed of 42 was used for reproducibility.

### Output Files

* `train.csv` – Training dataset.
* `validation.csv` – Validation dataset.
* `test.csv` – Testing dataset.

The resulting datasets will be used to train, tune, and evaluate the support ticket classification model.
