# Machine Learning for Public Procurement Transparency

## Predicting Procurement Categories from World Bank Notice Descriptions

**Author:** Constantine Akas  
**Institution:** African Leadership University  
**Project type:** Machine Learning and Deep Learning Classification

## 1. Project Overview

Public procurement is an important part of government operations because it supports the acquisition of goods, services, and infrastructure-related works. Organizing procurement information can help people analyze public purchasing activities and understand how procurement opportunities are distributed across categories.

This project develops and evaluates machine learning models that classify World Bank procurement notices using their textual bid descriptions. It compares traditional machine learning algorithms with a TensorFlow/Keras deep learning model to investigate how effectively different approaches predict the recorded procurement category.

The project aligns with my personal mission of contributing to stronger government institutions and infrastructure in Nigeria. It explores a practical application of machine learning to public procurement data. However, predicting procurement categories does not directly measure corruption, transparency, government effectiveness, or infrastructure quality.

## 2. Problem Statement

Procurement notices contain textual descriptions that provide information about the goods, works, or services being requested. Manually organizing large collections of notices can be time-consuming.

The research question is:

**How effectively can traditional machine learning and deep learning models predict procurement categories from the textual descriptions of World Bank procurement notices, and how well do the models perform on notices associated with Nigeria?**

This is a multiclass text-classification problem.

## 3. Dataset

The project uses the World Bank Procurement Notice dataset.

- **Source:** [World Bank Procurement Notice dataset](https://financesone.worldbank.org/d/DS00979)
- **Input feature:** Bid description text
- **Target variable:** Recorded procurement category
- **Classification task:** Four-class prediction
- **Geographic analysis:** Worldwide evaluation and a separate evaluation on Nigerian records

The four target classes are:

1. Consultant Services
2. Goods
3. Non-consulting Services
4. Works

The raw dataset contains more than 20,000 records, meeting the assignment's minimum dataset-size requirement. Records are filtered and prepared for modeling as documented in the notebook.

The dataset represents recorded procurement notices, and its labels reflect the categories in the source data. Model predictions should therefore be interpreted as category predictions, not independent assessments of procurement quality or integrity.

## 4. Methodology

### 4.1 Data preprocessing and feature engineering

The project prepares bid descriptions for text classification using TF-IDF (Term Frequency–Inverse Document Frequency). TF-IDF represents text numerically based on the importance of words and phrases within the dataset.

The traditional machine learning experiments compare unigram features with unigram-and-bigram features. The preprocessing configuration includes a minimum document frequency, a maximum feature limit, sublinear term-frequency scaling, lowercasing, and accent stripping.

Data splitting is performed by project group to reduce leakage between records belonging to the same project. The worldwide test set is kept separate from the training and validation sets for final evaluation.

### 4.2 Traditional machine learning

The notebook evaluates the following approaches:

- Majority-class baseline
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

Experiments investigate text-feature settings and class weighting. Class weighting is particularly relevant because the four procurement categories are not equally represented.

### 4.3 Deep learning

A neural text-classification model is implemented with TensorFlow/Keras. The architecture uses an embedding layer, SpatialDropout1D, GlobalAveragePooling1D, a regularized dense layer, dropout, and a four-class softmax output.

The notebook also explores regularization and early stopping to reduce overfitting and improve generalization. The input pipeline uses TensorFlow data-processing tools as documented in the notebook.

### 4.4 Evaluation

The models are assessed using:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Per-class precision, recall, and F1-score
- Confusion matrices
- One-vs-rest ROC-AUC
- Training and validation learning curves where applicable

Macro F1 gives equal weight to each class, while weighted F1 accounts for the number of records in each class. Reporting both helps reveal performance differences that overall accuracy may hide.

## 5. Results

The following are the recorded worldwide test-set results from the notebook.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 90.21% | 85.18% | 90.22% |
| Linear SVM | 90.79% | 84.92% | 90.46% |
| Regularized neural network | 88.27% | 83.25% | 88.65% |

Among these experiments, Linear SVM achieved the highest accuracy and weighted F1-score. Logistic Regression achieved the highest macro F1-score. The regularized neural network did not outperform the strongest traditional models on these three metrics.

The results also reveal a more difficult classification problem for Non-consulting Services, the least represented class. Its test-set F1-score was lower than those of the other three classes across all three models.

A separate evaluation on Nigerian records provides an additional perspective on geographic performance. The notebook reports lower overall and macro F1 scores for Nigerian records than for the worldwide test set. This difference motivates further investigation into class distribution, textual patterns, and geographic generalization.

These results are specific to the recorded dataset, preprocessing choices, model configurations, and evaluation splits. They do not establish that the models would perform equally well on all procurement systems or future notices.

## 6. Repository Contents

The repository contains the following project materials:

- `submission_ready_review_cleaned.ipynb` — the documented notebook containing data preparation, experiments, evaluation, and result interpretation.
- `README.md` — this project overview and documentation.
- `requirements.txt` — the Python dependencies required to reproduce the experiments.

Additional files may be added as the project is finalized.

## 7. Reproducing the Project

The notebook was developed for a Google Colab environment. To reproduce the experiments:

1. Clone or download this repository.
2. Obtain the dataset from the linked World Bank source, subject to its access and licensing terms.
3. Follow the data-loading instructions and file paths documented in the notebook.
4. Install the dependencies listed in `requirements.txt`.
5. Execute the notebook from beginning to end.
6. Compare the newly generated metrics with the results documented in the notebook.

Exact reproduction may depend on the Python and library versions, available hardware, random seeds, and model-training settings.

The raw dataset is not included in this GitHub repository. The project dataset and saved model artifacts will be provided through the separate Google Drive folder required for the assignment.

## 8. Limitations and Future Work

The dataset contains imbalanced target classes, particularly Non-consulting Services. Consequently, overall accuracy alone does not adequately describe performance across all categories.

The recorded category is used as the target label, so the model learns patterns in existing descriptions and labels. It may reproduce inconsistencies in those records and may perform differently on notices from other regions, time periods, or procurement systems.

Future work could investigate additional text representations, more advanced language models, temporal validation, further analysis of errors in the underrepresented class, and evaluation on independently collected procurement records. Such work would be necessary before considering practical deployment.

## 9. Project Links

- **GitHub repository:** https://github.com/codestantceasar/procurement-category-classification
- **Google Drive folder:** To be added after the folder is created and sharing permissions are checked.
- **Presentation video:** To be added after recording and uploading the presentation.

## 10. Acknowledgements

The project uses publicly sourced World Bank procurement data and open-source machine learning tools. Dataset attribution and applicable terms should be retained when redistributing or using the data.

This project was developed as part of a machine learning course at African Leadership University.
