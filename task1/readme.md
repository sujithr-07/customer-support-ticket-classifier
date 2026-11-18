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

# **TASK - 2**

## Identify 10 most confusing examples

The script saves:

* `<span data-d-streaming-word="">results/category_confusing_examples.csv</span>`
* `<span data-d-streaming-word="">results/priority_confusing_examples.csv</span>`

These examples are selected based on the lowest probability assigned to the actual class.

Add a short explanation for each example after reviewing the CSV files.

Example of possible explanations:

| Confusion                           | Possible reason                                              |
| ----------------------------------- | ------------------------------------------------------------ |
| Billing → Account Access           | Both mention account-related payments                        |
| Shipping → Product Inquiry         | Both mention product delivery                                |
| Technical Support → Account Access | Login problems can be technical issues                       |
| High → Critical                    | Similar urgency-related language                             |
| Medium → High                      | Customer describes an unresolved issue without clear urgency |

These are illustrative explanations, not measured errors from your dataset.


# TASK - 3

### Deployment Decision

1. DistilBERT can capture contextual information in customer support tickets and may perform better when ticket descriptions contain complex language.
2. TF-IDF with Logistic Regression provides a lightweight solution with faster inference and lower computational requirements.
3. DistilBERT requires more training time, memory, and computational resources, especially when deployed without a GPU.
4. If both models achieve similar validation and test Macro F1 scores, Logistic Regression is the more cost-effective deployment choice.
5. The final model will be selected based on test performance, inference speed, deployment cost, and classification reliability.
