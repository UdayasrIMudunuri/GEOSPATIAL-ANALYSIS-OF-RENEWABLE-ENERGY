import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
def BiomassPredict(state):
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro '], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('Biomass', axis=1)
    y = data['Biomass']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train individual models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    ann_model.fit(X_train, y_train)  

    # Make predictions with individual models
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_ann = ann_model.predict(X_test)  

    # Combine predictions using GradientBoostingRegressor as the meta-learner
    meta_X = np.column_stack((y_pred_rf, y_pred_xgb, y_pred_ann))
    meta_model = GradientBoostingRegressor(random_state=42)
    meta_model.fit(meta_X, y_test)

    # Make final predictions using the ensemble
    meta_X_test = np.column_stack((y_pred_rf, y_pred_xgb, y_pred_ann))
    y_pred_ensemble = meta_model.predict(meta_X_test)

    # Create a DataFrame for actual and predicted values
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ensemble})

    # Print actual and predicted values
    print(results_df)
    # Calculate metrics for the ensemble model
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    print(f'Ensemble Model Mean Squared Error (MSE): {mse_ensemble}')
    rmse_ensemble = np.sqrt(mse_ensemble)
    from sklearn.metrics import mean_absolute_percentage_error, r2_score

    # Calculate MAPE
    mape_ensemble = mean_absolute_percentage_error(y_test, y_pred_ensemble)

    # Calculate R2 score
    r2_ensemble = r2_score(y_test, y_pred_ensemble)

    # Print all metrics

    print(f'Ensemble Model Root Mean Squared Error (RMSE): {rmse_ensemble}')
    print(f'Ensemble Model Mean Absolute Percentage Error (MAPE): {mape_ensemble}')
    print(f'Ensemble Model R-squared (R2) Score: {r2_ensemble}')
    import matplotlib.pyplot as plt
    # Load the dataset
    newdata = pd.read_excel('predicted_values_all_features_2024.xlsx')
    #print(data)
    # Drop irrelevant columns
    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro '], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    newdata["MONTH"] = label_encoder.fit_transform(newdata["MONTH"])

    # Make predictions with individual models
    y_pred_rf = rf_model.predict(newdata)
    y_pred_xgb = xgb_model.predict(newdata)
    y_pred_ann = ann_model.predict(newdata)  

    # Combine predictions using GradientBoostingRegressor as the meta-learner
    #meta_X = np.column_stack((y_pred_rf, y_pred_xgb, y_pred_ann))
    #meta_model = GradientBoostingRegressor(random_state=42)
    #meta_model.fit(meta_X, y_test)

    # Make final predictions using the ensemble
    meta_X_test = np.column_stack((y_pred_rf, y_pred_xgb, y_pred_ann))
    y_pred_ensemble = meta_model.predict(meta_X_test)
    print("Bio Mass Prediction of 12 months of 2024")
    for x in y_pred_ensemble:
        print(x)

    return y_pred_ensemble
