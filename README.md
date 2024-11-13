[report_ml.pdf](https://github.com/user-attachments/files/17731045/report_ml.pdf)
# ML_project
Abstract 

Customer segmentation is a crucial marketing technique that allows businesses to tailor their strategies by grouping customers based on shared behaviors and purchasing patterns. In this project, we analyzed an E-commerce dataset using several machine learning models, including Support Vector Machines (SVM), Logistic Regression, k-Nearest Neighbors (k-NN), Decision Trees, and ensemble methods such as Random Forest, AdaBoost, and Gradient Boosting. Our goal was to develop a data-driven segmentation strategy to categorize customers into groups for better-targeted marketing. Using Recency, Frequency, and Monetary (RFM) metrics derived from the transactional data, we trained models to predict customer segments. After model evaluation, we found that ensemble methods like Gradient Boosting and Voting Classifier outperformed simpler models, demonstrating the effectiveness of combining multiple models. The outcome of this project shows how machine learning can enhance customer understanding and improve business strategies. 

1.Introduction 

In the competitive world of E-commerce, understanding customer behavior is essential for businesses to remain relevant and successful. Customer segmentation provides an effective way to categorize customers based on their purchasing behavior, allowing businesses to personalize marketing strategies and predict future behavior. This project uses a dataset from an E-commerce platform to segment customers using machine learning models based on Recency, Frequency, and Monetary (RFM) analysis. 
The objective of this project is to apply machine learning algorithms to classify customers into segments based on their purchasing patterns and optimize the segmentation process using predictive modeling techniques. The project covers data preprocessing, feature engineering, model building, and evaluation to identify the most accurate machine learning model for customer segmentation. 

2. Dataset Overview 
The dataset contains approximately 4,000 transactions from an E-commerce platform, including the following key features: 
● InvoiceNo: Unique identifier for each transaction. 
● StockCode: Unique code for each product. 
● Description: Product description. 
● Quantity: Number of units purchased in the transaction. 
● InvoiceDate: Date and time of the transaction. 
● UnitPrice: Price per unit of the product. 
● CustomerID: Unique identifier for each customer. 
● Country: Country where the customer resides.


Data Cleaning:
Before feature extraction, missing values (especially in the CustomerID column) were addressed, and negative quantities (indicating returns) were removed. After cleaning, the dataset was ready for analysis and feature engineering. 

Features used:
For this project, the Recency, Frequency, and Monetary (RFM) features were created from the raw transactional data. These three features were selected because they provide a strong foundation for customer segmentation, based on well-established principles in marketing analytics. 

Why only Recency, Frequency, and Monetary? 
The decision to focus on RFM features was based on the idea that these metrics capture key customer behaviors that are crucial for segmentation: 
● Recency: Measures how recently a customer made a purchase. Recency is important because recent buyers are more likely to engage again soon. 
● Frequency: Measures how often a customer makes purchases. Frequent buyers are generally loyal and can be targeted for future purchases. 
● Monetary: Measures how much money a customer has spent. High-value customers are typically more important to businesses in terms of revenue. 
These three dimensions (RFM) capture critical aspects of customer activity and purchasing behavior, which are particularly useful for customer segmentation tasks. The goal is to identify different customer groups based on how recently they engaged, how often they purchase, and how much they spend. 

How Were the RFM Features Calculated?
The RFM features(Recency, Frequency, and Monetary)were derived from the original 8 features in the dataset, with the following methodology:
Recency:
Calculated From: InvoiceDate
How: Recency is determined by calculating the number of days between the customer’s most recent transaction and a reference date, which is the latest date in the dataset.
Frequency:
Calculated From: InvoiceNo
How: Frequency represents the count of unique invoices for each customer, indicating the number of distinct transactions they made within the dataset period.

                             
Monetary Value:
Calculated From: Quantity and UnitPrice
  How: The Monetary value is computed by multiplying the quantity of items purchased by the price per unit, then summing these totals for each customer. 

Among the original dataset's 8 features—InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country—the following were crucial for the RFM analysis:
InvoiceDate, InvoiceNo, Quantity, and UnitPrice contributed directly to calculating Recency, Frequency, and Monetary values.
Features such as StockCode, Description, and Country were not directly used, as they pertain more to product specifics or location, which are less relevant for behavior-based customer segmentation using RFM.
CustomerID was used to group the data by individual customers for calculating each RFM feature.

Final Features for the Models 
The final set of features used for machine learning models are: 
● Recency: Number of days since the last purchase. 
● Frequency: Number of unique transactions. 
● Monetary Value: Total money spent by the customer. 
These three features were engineered from the relevant fields in the dataset, and they form the core of the input data used to train the machine learning models.
















3. Models Used and Justification 

3.1. Support Vector Machine (SVM) 
SVM is a powerful classification algorithm that works well in high-dimensional spaces and is effective for non-linear data separation. It uses a kernel trick to transform data into a higher-dimensional space where classes can be linearly separated. 
● Reason: SVM is ideal for complex, non-linear problems where other models may struggle to separate the data effectively. 

3.2. Logistic Regression 
Logistic Regression is a simple, interpretable model for binary and multi-class classification. It predicts the probability of a class based on input features using a logistic function. 
● Reason: It serves as a baseline model to compare performance with more complex algorithms. 

3.3. k-Nearest Neighbors (k-NN) 
k-NN is a distance-based algorithm that classifies data points based on the majority class of their nearest neighbors. It is non-parametric and does not make assumptions about the underlying data distribution. 
● Reason: k-NN helps explore how well a simple distance-based method performs for customer segmentation. 

3.4. Decision Tree 
Decision Trees are rule-based classifiers that split the data into branches based on feature importance. They are easy to interpret and visualize. 

● Reason: Decision Trees allow us to explore feature importance and handle both categorical and continuous data. 

3.5. Random Forest 
Random Forest is an ensemble method that builds multiple decision trees on different subsets of the data and averages their predictions for better accuracy. 
● Reason: It reduces overfitting and improves accuracy by leveraging multiple decision trees.
 
3.6. AdaBoost 
AdaBoost combines weak learners into a strong classifier by adjusting the weights of misclassified samples to improve subsequent predictions.
● Reason: AdaBoost improves performance by focusing on difficult-to-classify instances. 

3.7. Gradient Boosting 
Gradient Boosting builds models sequentially by minimizing the errors of the previous models through gradient descent optimization. 
● Reason: Gradient Boosting is highly effective for improving accuracy, though computationally expensive. 

3.8. Voting Classifier 
The Voting Classifier aggregates the predictions from multiple models (SVM, Logistic Regression, Random Forest) to form a majority-vote decision, leveraging the strengths of each model. 
● Reason: By combining different models, the Voting Classifier offers more robust and accurate predictions. 



























4. Methodology 
The methodology follows a structured process of data preparation, feature engineering, model implementation, and evaluation. 
4.1. Data Preprocessing 
● Handling Missing Data: Removed rows with missing CustomerID values. 
● Outlier Removal: Transactions with negative quantities were excluded to avoid skewing the data. 

4.2. Feature Engineering 
Derived the following features for segmentation: 
● Recency: Number of days since the last purchase. 
● Frequency: Number of unique transactions per customer. 
● Monetary Value: Total money spent by the customer. 

4.3. Model Training and Tuning 
Each model was trained using the engineered RFM features.

4.4. Evaluation 
Models were evaluated using accuracy, precision, recall, F1-score, and confusion matrices to assess performance on the test set. Cross-validation was used to avoid overfitting. 

5. Results 

Accuracy
Precision
Recall
F1-Score


SVC
0.971198
1.000000
0.945295
0.971879
Logistic Regression
1.000000
1.000000
1.000000
1.000000
k-NN
0.997696
0.997812
0.997812
0.997812
Decision Tree
0.998848
1.000000
0.997812
0.998905
Random Forest
0.998848
1.000000
0.997812
0.998905
AdaBoost
0.998848
1.000000
0.997812
0.998905
Gradient Boosting
0.998848
1.000000
0.997812
0.998905



6. Learning Outcome 
6.1. Skills Used 
● Data Preprocessing: Cleaning, handling missing values, outlier removal. 
● Feature Engineering: Creating RFM features for customer segmentation.
● Machine Learning: Implementing classification models and ensemble techniques.
● Model Evaluation: Using performance metrics to assess model effectiveness. 
6.2. Tools Used 
● Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook.
6.3. Dataset Used 
● The dataset contains customer transactions, which were processed to derive Recency, Frequency, and Monetary Value for segmentation. 

7. Conclusion 
Among the models tested, Logistic Regression achieved perfect scores in accuracy, precision, recall, and F1-score, making it the best-performing model. However, the 100% accuracy suggests a potential risk of overfitting, which could limit its generalizability to unseen data. Models like k-NN, Decision Tree, Random Forest, AdaBoost, and Gradient Boosting also performed exceptionally well with high accuracy (~99.88%) and balanced metrics, making them strong candidates for real-world applications. These models are suitable for customer segmentation, churn prediction, and targeted marketing strategies, offering reliable predictions for varied customer behaviors. Despite the impressive performance, the Logistic Regression model's results may not be entirely realistic due to overfitting. Future work should involve validating models on separate test sets, applying cross-validation, and exploring more advanced techniques such as neural networks. These steps would help improve model robustness and ensure better generalization in business scenarios. 










8. References 
● ChatGPT: For assisting with explanations and methodology structuring. 
   https://www.kaggle.com/code/fabiendaniel/customer-segmentation/notebook

Colab link: 
https://colab.research.google.com/drive/1UHKRsqT6WZNnxzLfZcNgl9hILq7uy2nB?usp=sharing
Github repository: 
https://github.com/Hema-hash/ML_project
