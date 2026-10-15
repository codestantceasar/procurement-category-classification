# Machine Learning for Public Procurement Transparency

## Predicting Procurement Categories from World Bank Notice Descriptions

**Author:** Constantine Akas  
**Institution:** African Leadership University  
**Project type:** Machine Learning and Deep Learning Classification  
**Task:** Multiclass Text Classification

## 1. Project Overview

Public procurement supports the acquisition of goods, services, and infrastructure-related works. Organizing procurement information can help people search, analyze, and understand large collections of public purchasing notices.

This project develops and evaluates machine learning models that predict the recorded procurement category of World Bank procurement notices using their textual bid descriptions. It compares traditional machine learning approaches with a TensorFlow/Keras neural network to investigate their classification performance.

The project aligns with my personal mission of contributing to stronger government institutions and infrastructure in Nigeria. It explores how machine learning can help organize procurement information. However, predicting procurement categories does not directly measure corruption, transparency, government effectiveness, or infrastructure quality.

## 2. Problem Statement

Procurement notices contain textual descriptions of the goods, works, and services being requested. Automatically categorizing large collections of notices could support information retrieval, indexing, and quality-control workflows.

The main research question is:

**How effectively can traditional machine learning and deep learning models predict procurement categories from World Bank procurement notice descriptions, and how do the models perform on notices associated with Nigeria?**

This is a four-class text-classification problem.

## 3. Dataset

The project uses the World Bank Procurement Notice dataset.

- **Source:** [World Bank Procurement Notice dataset](https://financesone.worldbank.org/d/DS00979)
- **Input feature:** Procurement notice bid-description text
- **Target variable:** Recorded procurement category
- **Classification task:** Four-class prediction
- **Geographic analysis:** Worldwide evaluation and a separate evaluation of Nigerian records

The four target classes are:

1. Consultant Services
2. Goods
3. Non-consulting Services
4. Works

### Dataset preparation

The original dataset contained **421,584 records**. After data cleaning and preparation, **384,722 records** remained for the modeling workflow.

The cleaned dataset contains notices from all four target categories. The classes are imbalanced, with Non-consulting Services being the least represented category.

The data was divided into training, validation, and test sets using project-grouped splitting. This approach helps prevent notices associated with the same project from appearing across multiple partitions.

| Dataset partition | Records |
|---|---:|
| Training | 276,855 |
| Validation | 47,985 |
| Test | 59,882 |
| **Total** | **384,722** |

The target labels reflect the categories recorded in the source data. Model predictions should therefore be interpreted as category predictions, not independent assessments of procurement quality or integrity.

## 4. Methodology

### 4.1 Data preprocessing and feature engineering

The traditional machine learning pipeline uses Term Frequency–Inverse Document Frequency (TF-IDF) to convert notice descriptions into numerical text features.

The experiments investigate unigram and unigram-plus-bigram representations. The preprocessing configuration includes lowercasing, accent stripping, sublinear term-frequency scaling, minimum document frequency, and a maximum feature limit.

Project-grouped splitting is used to reduce the risk of data leakage between notices associated with the same project. The held-out test set is reserved for final model evaluation.

### 4.2 Traditional machine learning

The notebook evaluates the following approaches:

- Majority-class baseline
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

The experiments compare text-feature configurations and investigate class weighting. Class weighting is relevant because the four procurement categories are not equally represented.

### 4.3 Deep learning

A neural text-classification model is implemented using TensorFlow/Keras. The architecture includes:

- Text input and an embedding layer
- SpatialDropout1D
- GlobalAveragePooling1D
- A regularized dense layer
- Dropout
- A four-class softmax output layer

The notebook also investigates regularization and early stopping to improve generalization and reduce overfitting.

Training and validation accuracy and loss curves are used to examine the learning behavior of the neural network.

### 4.4 Evaluation metrics

The models are evaluated using:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Per-class precision, recall, and F1-score
- Confusion matrices
- One-versus-rest ROC-AUC
- Training and validation learning curves

Macro F1 gives equal weight to each class, while weighted F1 accounts for the number of examples in each class. Reporting both helps reveal differences that overall accuracy may hide.

## 5. Results

### 5.1 Worldwide held-out test results

The following table summarizes the recorded worldwide held-out test results.

| Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC (OvR) | Weighted ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|
| Balanced TF-IDF + Logistic Regression | 90.23% | 85.19% | 90.24% | 0.9679 | 0.9823 |
| TF-IDF + Linear SVM | 90.69% | 84.64% | 90.31% | 0.9630 | 0.9804 |
| Regularized Keras neural network | 87.89% | 82.85% | 88.43% | 0.9600 | 0.9770 |

Linear SVM achieved the highest recorded test accuracy and weighted F1-score. Logistic Regression achieved the highest macro F1-score and macro and weighted ROC-AUC among the three final models.

The neural network performed below the two traditional models on the reported test metrics. These findings indicate that TF-IDF representations combined with linear classifiers were effective for this dataset and experimental configuration. They do not establish that traditional models will always outperform neural networks.

### 5.2 Category-level performance

Non-consulting Services was the most difficult category to classify.

In the recorded test classification report for the regularized neural network, this category achieved:

- **Precision:** 0.50
- **Recall:** 0.68
- **F1-score:** 0.58

By comparison, the neural network achieved F1-scores of 0.92 for Consultant Services, 0.91 for Goods, and 0.90 for Works.

The results demonstrate why accuracy alone is insufficient for evaluating imbalanced multiclass problems. Differences in class representation, overlapping terminology, generic descriptions, and inconsistencies in recorded labels may contribute to the classification challenge.

The class-weighting experiment also showed a precision–recall trade-off for Non-consulting Services. In the recorded validation comparison, recall increased from 0.52 to 0.74 while precision decreased from 0.80 to 0.63. Class weighting therefore changed the error profile rather than improving every metric simultaneously.

### 5.3 Nigeria-specific evaluation

A separate evaluation examined **2,707 held-out notices associated with Nigeria**.

| Model | Evaluation set | Accuracy | Macro F1 |
|---|---|---:|---:|
| Logistic Regression | Worldwide test set | 90.23% | 85.19% |
| Logistic Regression | Nigeria subset | 87.81% | 78.97% |
| Linear SVM | Worldwide test set | 90.69% | 84.64% |
| Linear SVM | Nigeria subset | 88.62% | 80.70% |

Both traditional models recorded lower accuracy and macro F1 on the Nigerian subset than on the worldwide test set.

Non-consulting Services remained challenging in the Nigeria-specific evaluation. Logistic Regression achieved precision of 0.35, recall of 0.71, and F1 of 0.47. Linear SVM achieved precision of 0.46, recall of 0.62, and F1 of 0.53.

These results motivate further investigation into category distributions, vocabulary, notice templates, and other potential differences between geographic subsets. The evaluation does not establish why the performance gap occurred, and it should not be interpreted as evidence that Nigerian procurement is less transparent or more corrupt.

### 5.4 Neural-network diagnostics

The notebook includes training and validation learning curves to help assess the neural network's learning behavior.

In the recorded initial experiment, training accuracy increased from 91.94% in the first epoch to 96.63% in the third epoch, while validation accuracy remained around 87.5–87.9% and validation loss increased from 0.3830 to 0.4291. This pattern is consistent with possible overfitting.

The second neural-network experiment used a smaller architecture and additional regularization. It produced a modest improvement in validation macro F1. These results suggest that regularization helped to some extent but did not close the performance gap with the strongest traditional models.

All reported results should be checked against the corresponding notebook outputs before final submission.

## 6. Repository Contents

The repository is intended to contain the following materials:

- `README.md` — project overview, methodology, results, and reproduction instructions.
- `requirements.txt` — Python dependencies for the project.
- Jupyter notebook — data preparation, model experiments, evaluation, visualizations, and result interpretation.

Update the notebook filename listed here to match the actual file committed to the repository.

## 7. Reproducing the Project

The notebook was developed for Google Colab. To reproduce the experiments:

1. Clone or download the repository.
2. Obtain the dataset from the linked World Bank source, subject to its access and licensing terms.
3. Upload or make the dataset available in Google Colab.
4. Follow the notebook's data-loading instructions and file paths.
5. Install the dependencies listed in `requirements.txt`.
6. Execute the notebook cells in order, including the neural-network training and learning-curve cells.
7. Review the generated metrics, classification reports, confusion matrices, ROC curves, and learning curves.
8. Compare the newly generated outputs with the results reported in this README.

Exact reproduction may depend on Python and library versions, hardware, random seeds, data preparation, and model-training settings.

The raw dataset is not included in this repository. Dataset files and any required supporting artifacts are intended to be provided through a separate Google Drive folder for the assignment. Add the folder link once it has been created and its sharing permissions have been verified.

## 8. Limitations and Future Work

Several limitations affect interpretation of the results.

First, the dataset consists of notices published through the World Bank system and is not necessarily representative of all procurement activities in Nigeria or worldwide.

Second, the target is the category already recorded in the source data. The models may learn inconsistencies in existing labels rather than independently verified categories.

Third, the input is limited to notice descriptions. The models do not directly assess complete tender documents, contract execution, procurement outcomes, or audit-confirmed irregularities.

Fourth, the class imbalance makes Non-consulting Services harder to classify. Increasing recall through class weighting reduces precision, so model selection depends on the relative costs of missed cases and false alarms.

Fifth, the Nigeria-specific subset is smaller than the worldwide test set and may have a different category distribution. The observed performance difference is descriptive and does not identify its cause.

Finally, the reviewed notebook has not been executed end-to-end after editing. The displayed outputs are retained from earlier experiments, and a fresh run may produce different values depending on software versions, hardware, and training settings. A clean end-to-end run is necessary before claiming full reproducibility.

Future work could include:

1. Executing the notebook in a fresh Colab runtime and verifying all outputs.
2. Recording software and dependency versions.
3. Reporting class counts for each evaluation subset.
4. Using repeated project-grouped splits or confidence intervals.
5. Evaluating additional geographic subsets.
6. Reviewing false positives and false negatives for Non-consulting Services.
7. Investigating targeted data cleaning and alternative class-weighting strategies.
8. Testing transformer-based text models if computational resources permit.

If the long-term objective expands to procurement risk analysis, redesign the target around a clearly defined, independently validated risk indicator rather than treating category predictions as evidence of corruption.

## 9. Project Links

- **GitHub repository:** https://github.com/codestantceasar/procurement-category-classification
- **Google Drive folder:** Add the verified link after creating the folder and checking its sharing permissions.
- **Presentation video:** Add the link after recording and uploading the presentation.

## 10. Acknowledgements

This project uses World Bank procurement data and open-source machine learning tools. Dataset attribution and applicable terms should be retained when redistributing or using the data.

The project was developed as part of a machine learning course at African Leadership University.
