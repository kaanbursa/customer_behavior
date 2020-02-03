# Santander Bank Targeted Advertisement

## Summary

![](./front_page.png)

We wanted to create a proudct for Santander Bank which would increase their marketing efficiency. Santander Bank offer variety of products and has collected millions of data from old customers. We get our dataset from (Santander Bank's Kaggle challenge )[https://www.kaggle.com/c/santander-product-recommendation] . Our goal was to train a model which would recommend current account to upcoming customers.

## Notebooks

1. EDA is our notebook where we clean data and explore the dataset categorize try to extract meaning from visualization and informations.
2. Model.ipnby notebook is our notebook where we run different models to find the best predictor models




## Conclusion

We run our models on Google Colab which took several hours to train multiple models with multiple hyperparameters. Finally XGBoost performed better than all other models with best 'roc_auc_score'. After having a model we picked a threshold value for our model which is probabilistic. We fine tuned the hyperparameters and used the XGBoost model and test it on our test set.

So our model has the accuracy of ----- which means that with our model Santander Bank can have targeted advertisement strategy using our model.

