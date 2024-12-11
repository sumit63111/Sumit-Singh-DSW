# Sumit-Singh-DSW

# Loan Approval Prediction

This repository contains a project for predicting loan approvals based on customer data. The project involves Exploratory Data Analysis (EDA) to uncover insights and relationships within the data and the implementation of three machine learning models: **Random Forest**, **Logistic Regression**, and **AdaBoost**. The model with the highest accuracy is selected for deployment.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Workflow](#project-workflow)
3. [Dataset Description](#dataset-description)
4. [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
5. [Machine Learning Models](#machine-learning-models)
6. [Model Selection](#model-selection)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [Dependencies](#dependencies)
10. [Conclusion](#conclusion)

---

## Introduction

The goal of this project is to predict whether a loan application will be approved based on various applicant details. The prediction can assist financial institutions in making informed decisions efficiently.

---

## Project Workflow

1. Data preprocessing: Handle missing values, encode categorical variables, and scale numerical features.
2. Exploratory Data Analysis: Identify patterns and relationships in the dataset.
3. Model training: Train three models (Random Forest, Logistic Regression, AdaBoost).
4. Model evaluation: Compare model performance metrics.
5. Model selection: Select the model with the highest accuracy.

---

## Dataset Description

The dataset contains the following features:

- **ApplicantIncome**: Income of the applicant.
- **LoanAmount**: Loan amount requested.
- **Loan_Amount_Term**: Term of the loan.
- **Credit_History**: Credit history of the applicant.
- **Loan_Status**: Target variable (1: Approved, 0: Not Approved).

---

## EDA (Exploratory Data Analysis)

The EDA phase involves:

1. Distribution analysis of numerical and categorical features.
2. Identifying correlations between features and the target variable.
3. Handling outliers and missing data.
4. Visualizing important patterns using plots such as histograms, box plots, and scatter plots.

---

## Machine Learning Models

The following models were implemented:

1. **AdaBoost**: Achieved an accuracy of **77%**
2. **Logistic Regression**: Achieved an accuracy of **74%**.
3. **Random Forest**: It ranges between **76 80%**

---

## Model Selection

A separate file, `model_selection.ipynb`, is included in this repository. This script automatically selects the model with the highest accuracy, 
As their are minor differnce in a accuray of random forest and adaboost

so when randomforest accuracy is 76 then it will select the **Adaboost** as preferred model.
if its greater than 76% then it will select **Random forest**.
