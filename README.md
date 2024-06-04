
# Titanic Survival Prediction

## Project Overview
This project uses machine learning to predict the survival of passengers on the Titanic based on various features. The model is trained on the Titanic dataset, which includes information such as passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, fare, and port of embarkation.

## Dataset Overview
The dataset used in this project is the [Kaggle Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset). It contains information about the passengers aboard the Titanic.

- **Number of Instances:** 891
- **Number of Features:** 12
  - **PassengerId:** Unique ID for each passenger.
  - **Pclass:** Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
  - **Name:** Name of the passenger.
  - **Sex:** Gender of the passenger.
  - **Age:** Age of the passenger.
  - **SibSp:** Number of siblings/spouses aboard the Titanic.
  - **Parch:** Number of parents/children aboard the Titanic.
  - **Ticket:** Ticket number.
  - **Fare:** Passenger fare.
  - **Cabin:** Cabin number.
  - **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
  - **Survived:** Target variable (0 = No, 1 = Yes).

## Project Structure

The project is organized as follows:

1. **Data Preprocessing:**
   - Handling missing values.
   - Encoding categorical variables.

2. **Exploratory Data Analysis (EDA):**
   - Visualizing the relationship between features and survival.

3. **Feature Engineering:**
   - Creating new features to improve model performance.

4. **Model Building and Evaluation:**
   - Training various machine learning models.
   - Evaluating model performance using metrics such as accuracy and ROC-AUC.

5. **Prediction:**
   - Using the trained model to predict the survival of new passengers.

6. **Visualization:**
   - Presenting the results through graphs and charts for better understanding and interpretation.

## Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Results

- The final model's performance is summarized in terms of accuracy, precision, recall, and F1 score.
- Confusion matrix visualizations provide insights into the classification results.

## Conclusion

This project demonstrates the process of building a machine learning model to predict survival on the Titanic. It covers data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and prediction.

## Usage

To run this project, ensure you have the required libraries installed. You can install the dependencies using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run the Jupyter notebook to execute the code step-by-step.

## Acknowledgements

- The dataset is provided by [Kaggle](https://www.kaggle.com/c/titanic).

