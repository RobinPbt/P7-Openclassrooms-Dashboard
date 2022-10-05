# P7-Openclassrooms-Dashboard
This repository has been created in the context of the 7th project of my Data Scientist training with Openclassrooms. 
The goal of this project is to use datas from a former Kaggle competition (https://www.kaggle.com/c/home-credit-default-risk/data) to build a classification model. 
The model is a binary classification with a positive class corresponding to a customer with loan repayment difficulties. 
The model must take into consideration a business optimization considering false negatives have a bigger impact than a false positive on bank revenues. 
Then the model is deployed throught an API to make predictions and a dashboard to display results visually.

This repository concerns the dashboard application. This application has been built with Dash library and is inspired by the following template : https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-manufacture-spc-dashboard

The other repositories about this project can be found on the following links :
- API : https://github.com/RobinPbt/P7-Openclassrooms-API
- Modeling : https://github.com/RobinPbt/P7-Openclassrooms

The dashboard application can be found on the following link : https://p7-openclassrooms-dash.herokuapp.com/

# Screenshots

![image](https://user-images.githubusercontent.com/104992181/193087389-f8685ef4-d192-4b79-840a-cee0423a798b.png)
![image](https://user-images.githubusercontent.com/104992181/193087412-38f165e0-f299-4f8a-baaf-361d19a7047e.png)
![image](https://user-images.githubusercontent.com/104992181/193087429-8ed3fea7-9471-4785-bbcf-d7553494723a.png)

# Run locally

In order to run locally, you must run the API first (see https://github.com/RobinPbt/P7-Openclassrooms-API) and then download the files and modify app.py:
- line 280 : replace the PREDICT_URI variable with the URL of your local API (http://127.0.0.1:XXXX/api/predict-existing/)
- line 471 : replace the PREDICT_URI variable with the URL of your local API (http://127.0.0.1:XXXX/api/predict-new/) 

You can then run the dashboard locally with the command "python app.py", your dashboard will be launched on http://127.0.0.1:XXXX/ (replace XXXX by the corresponding port)
