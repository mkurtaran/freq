from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import lightgbm as lgb
from typing import Any, Dict
import pandas as pd

class my_model(BaseRegressionModel):
    """
    User created model.
    This model uses a classification logic inside a regression model structure
    to bypass version-specific import issues.
    """

    def fit(self, data_kitchen: FreqaiDataKitchen, **kwargs):
        """
        This function is called to fit the model.
        """
        # Extract data from the data_kitchen
        X = data_kitchen.X_train
        y = data_kitchen.y_train.values.ravel()

        # Define and train the LightGBM classifier
        lgb_clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        lgb_clf.fit(X, y)

        return lgb_clf

    def predict(self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs) -> pd.DataFrame:
        """
        This function is called to make predictions.
        """
        # Use the trained model from the data_kitchen to make predictions
        model = dk.trained_model
        features = dk.X_pred

        # Predict probabilities
        probabilities = model.predict_proba(features)

        # We are interested in the probability of the positive class (1)
        predictions = probabilities[:, 1]

        # Add the predictions to the dataframe
        unfiltered_df['&s-prediction'] = pd.Series(predictions, index=features.index)

        return unfiltered_df
