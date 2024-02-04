
# Text Classification Case Study

This project aims to explore and implement text classification techniques on a real-world dataset. Text classification involves assigning predefined categories or labels to a given set of textual data. In this case study, we focus on applying text classification to a specific dataset to gain insights and build a predictive model.

## Dataset

For this case study we use [Zenodo E commerce text dataset](https://zenodo.org/records/3355823).

Here's a description provided by Zenodo:

> This is the classification based E-commerce text dataset for 4 categories - "Electronics", "Household", "Books" and "Clothing & Accessories", which almost cover 80% of any E-commerce website. 

> The dataset is in ".csv" format with two columns - the first column is the class name and the second one is the datapoint of that class. The data point is the product and description from the e-commerce website.

## Task setting

Imagine we own a E-commerce website. On this website the seller can upload the descriptions of the items they want to sell. They also have to choose items' categories manually which may slow the seller down.

Our task is to automate the choice of categories based on item description.

However, the wrong automated choice may lead to losses in sales, therefore we may choose not to set an automated label if we're not sure.

## Techniques used and compared

In this case study we implement and compare 4 text classification techniques:

- Baseline: Bag of Words + Logistic regression
- GRU
- LSTM using pretrained embedding layer
- Fine-tuned BERT

We compare them by:
- classification quality
- inference time keeping in mind that we would want to use the model in production environment in our imaginary task
- the percentage of precise auto verdicts - i.e. how often the seller would have to interfere if the model is not sure.
