import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset

def generate(state):
    print("State name is ",state)
    global data
    df = pd.read_excel(state)
    print("State file path:", state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'air_temp']
    data = df[columns_to_keep]  # Use df instead of data

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('air_temp', axis=1)
    y = data['air_temp']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_temperatures_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_temperatures_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, temperature in predicted_temperatures_2024.items():
        print(f"Predicted air temperature for {month} 2024: {temperature}")



    #2
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'albedo']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('albedo', axis=1)
    y = data['albedo']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_albedo_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_albedo_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, albedo in predicted_albedo_2024.items():
        print(f"Predicted albedo for {month} 2024: {albedo}")


    #3
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'clearsky_dhi']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('clearsky_dhi', axis=1)
    y = data['clearsky_dhi']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_clearsky_dhi_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_clearsky_dhi_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, clearsky_dhi in predicted_clearsky_dhi_2024.items():
        print(f"Predicted clearsky_dhi for {month} 2024: {clearsky_dhi}")


    #4
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'clearsky_dni']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('clearsky_dni', axis=1)
    y = data['clearsky_dni']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_clearsky_dni_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_clearsky_dni_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, clearsky_dni in predicted_clearsky_dhi_2024.items():
        print(f"Predicted clearsky_dni for {month} 2024: {clearsky_dni}")

    #5

    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'clearsky_gti']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('clearsky_gti', axis=1)
    y = data['clearsky_gti']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_clearsky_gti_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_clearsky_gti_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, clearsky_gti in predicted_clearsky_gti_2024.items():
        print(f"Predicted clearsky_gti for {month} 2024: {clearsky_gti}")

    #6
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'cloud_opacity']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('cloud_opacity', axis=1)
    y = data['cloud_opacity']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_cloud_opacity_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_cloud_opacity_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, cloud_opacity in predicted_cloud_opacity_2024.items():
        print(f"Predicted cloud_opacity for {month} 2024: {cloud_opacity}")

    #7

    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'dni']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('dni', axis=1)
    y = data['dni']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_dni_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_dni_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, dni in predicted_dni_2024.items():
        print(f"Predicted dni for {month} 2024: {dni}")

    #8

    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'ghi']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('ghi', axis=1)
    y = data['ghi']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_ghi_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_ghi_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, ghi in predicted_ghi_2024.items():
        print(f"Predicted ghi for {month} 2024: {ghi}")

    #9
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'gti']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('gti', axis=1)
    y = data['gti']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_gti_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_gti_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, gti in predicted_gti_2024.items():
        print(f"Predicted gti for {month} 2024: {gti}")


    #10
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'precipitation_rate']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('precipitation_rate', axis=1)
    y = data['precipitation_rate']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_precipitation_rate_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_precipitation_rate_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, precipitation_rate in predicted_precipitation_rate_2024.items():
        print(f"Predicted precipitation_rate for {month} 2024: {precipitation_rate}")
        

    #11
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'relative_humidity']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('relative_humidity', axis=1)
    y = data['relative_humidity']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_relative_humidity_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_relative_humidity_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, relative_humidity in predicted_relative_humidity_2024.items():
        print(f"Predicted relative_humidity for {month} 2024: {relative_humidity}")


    #12
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'surface_pressure']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('surface_pressure', axis=1)
    y = data['surface_pressure']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_surface_pressure_2024 = {}
    print(type(predicted_surface_pressure_2024))

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_surface_pressure_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, surface_pressure in predicted_surface_pressure_2024.items():
        print(f"Predicted surface_pressure for {month} 2024: {surface_pressure}")

    #13
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    columns_to_keep = ['YEAR', 'MONTH', 'Latitude', 'Longitude', 'wind_speed_100m']
    data = data[columns_to_keep]

    #data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro ',], axis=1, inplace=True)

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('wind_speed_100m', axis=1)
    y = data['wind_speed_100m']

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
    # Assuming you have a function to extract meteorological features for a specific month in 2024
    def extract_monthly_features(year, month):
        # Implement logic to extract features for a specific month in 2024
        # For now, let's assume the values are similar to the last available month's data
        last_month_data = X.iloc[-1]
        monthly_data = last_month_data.copy()
        monthly_data['MONTH'] = label_encoder.transform([month])
        monthly_data['YEAR'] = year
        return monthly_data.values.reshape(1, -1)

    # Predict air temperatures for all 12 months of 2024
    predicted_wind_speed_100m_2024 = {}

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in months:
        monthly_features = extract_monthly_features(2024, month)

        # Make predictions for the month
        monthly_rf = rf_model.predict(monthly_features)
        monthly_xgb = xgb_model.predict(monthly_features)
        monthly_ann = ann_model.predict(monthly_features)

        # Combine predictions using the meta-learner
        monthly_meta_X = np.column_stack((monthly_rf, monthly_xgb, monthly_ann))
        monthly_prediction = meta_model.predict(monthly_meta_X)

        predicted_wind_speed_100m_2024[month] = monthly_prediction[0]

    # Print predicted air temperatures for all months of 2024
    for month, wind_speed_100m in predicted_wind_speed_100m_2024.items():
        print(f"Predicted wind_speed_100m for {month} 2024: {wind_speed_100m}")
    predicted_values_2024 = {
        'air_temp': predicted_temperatures_2024,
        'albedo': predicted_albedo_2024,
        'clearsky_dhi': predicted_clearsky_dhi_2024,
        'clearsky_dni': predicted_clearsky_dni_2024,
        'clearsky_gti': predicted_clearsky_gti_2024,
        'cloud_opacity': predicted_cloud_opacity_2024,
        'dni': predicted_dni_2024,
        'ghi': predicted_ghi_2024,
        'gti': predicted_gti_2024,
        'precipitation_rate': predicted_precipitation_rate_2024,
        'relative_humidity': predicted_relative_humidity_2024,
        'surface_pressure': predicted_surface_pressure_2024,
        'wind_speed_100m': predicted_wind_speed_100m_2024
    }

    # Convert the dictionary to a DataFrame
    predicted_df = pd.DataFrame(predicted_values_2024)
    year_2024 = [2024] * 12

    # Add the 'YEAR' column to the DataFrame
    predicted_df.insert(0, 'YEAR', year_2024)
    predicted_df.insert(1, 'MONTH', months)
    predicted_df['MONTH'] = months
    top_12_rows = data['Latitude'].head(12)
    newLatitude=top_12_rows.tolist()
    top_12_rows = data['Longitude'].head(12)
    newLongitude=top_12_rows.tolist()



    predicted_df.insert(2, 'Latitude', newLatitude)
    predicted_df.insert(3, 'Longitude', newLongitude)

    # Save the DataFrame to an Excel file
    predicted_df.to_excel('predicted_values_all_features_2024.xlsx', index=False)

    print("All predicted values with additional information for 2024 have been saved to predicted_values_all_features_2024.xlsx file.")


