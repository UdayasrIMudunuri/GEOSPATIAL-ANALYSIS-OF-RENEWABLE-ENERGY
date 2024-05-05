import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
fres=[]

def stateImp(state):
    state="data/"+state+".xlsx"
    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    data.drop(['Name of State/UT', 'Solar', 'Wind ', 'Small Hydro '], axis=1, inplace=True,errors='ignore')

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()    
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('Biomass ', axis=1,errors='ignore')
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
    #fres.append(mse_ensemble)

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
    fres.append(rmse_ensemble)
    #fres.append(mape_ensemble)
    fres.append(r2_ensemble)

    # Group the data by year and calculate the mean wind power generation for each year
    wind_yearly_mean = data.groupby('YEAR')['Biomass'].mean().reset_index()
    # Create a DataFrame for actual and predicted values
    #results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ensemble})

    # Sort DataFrame by 'Actual' values for better visualization
    results_df.sort_values(by='Actual', inplace=True)
    
    
    # Group the data by year and calculate the mean Biomass power generation for each year
    wind_yearly_mean = data.groupby('YEAR')['Biomass'].mean().reset_index()

    # Plot the trend of Biomass power generation over the years
    plt.figure(figsize=(10, 6))
    plt.plot(wind_yearly_mean['YEAR'], wind_yearly_mean['Biomass'], marker='o', linestyle='-')
    plt.title('Trend of Biomass Power Generation Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Biomass Power Generation')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig("static/biomass.jpg")
    #plt.show()
    Solar(state)

    return fres    

def Solar(state):
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    data.drop(['Name of State/UT', 'Wind ', 'Biomass', 'Small Hydro '], axis=1, inplace=True,errors='ignore')

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])
    data.columns = data.columns.str.strip()

    # Split features and target variable
    X = data.drop('Solar ',axis=1,errors='ignore')
    y = data['Solar']

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
    #fres.append(mse_ensemble)
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
    fres.append(rmse_ensemble)
    #fres.append(mape_ensemble)
    fres.append(r2_ensemble)

    # Group the data by year and calculate the mean Solar  power generation for each year
    wind_yearly_mean = data.groupby('YEAR')['Solar'].mean().reset_index()

    # Plot the trend of Solar power generation over the years
    plt.figure(figsize=(10, 6))
    plt.plot(wind_yearly_mean['YEAR'], wind_yearly_mean['Solar'], marker='o', linestyle='-')
    plt.title('Trend of Solar Power Generation Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Solar Power Generation')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig("static/Solar.jpg")
    #plt.show()

    SmallHydro(state)


def SmallHydro(state):

    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    data.drop(['Name of State/UT', 'solar Power', 'Wind ', 'Biomass'], axis=1, inplace=True,errors='ignore')

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('Small Hydro ', axis=1)
    y = data['Small Hydro ']

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
    #fres.append(mse_ensemble)
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
    fres.append(rmse_ensemble)
    #fres.append(mape_ensemble)
    fres.append(r2_ensemble)

    # Group the data by year and calculate the mean Small Hydro   power generation for each year
    wind_yearly_mean = data.groupby('YEAR')['Small Hydro '].mean().reset_index()

    # Plot the trend of Small Hydro  power generation over the years
    plt.figure(figsize=(10, 6))
    plt.plot(wind_yearly_mean['YEAR'], wind_yearly_mean['Small Hydro '], marker='o', linestyle='-')
    plt.title('Trend of Small Hydro  Power Generation Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Small Hydro  Power Generation')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig("static/SmallHydro.jpg")
    #plt.show()
    wind(state)


def wind(state):


    # Load the dataset
    data = pd.read_excel(state)
    #print(data)
    # Drop irrelevant columns
    data.drop(['Name of State/UT', 'solar Power', 'Small Hydro ', 'Biomass'], axis=1, inplace=True,errors='ignore')

    # Label encode 'MONTH' column
    label_encoder = LabelEncoder()
    data["MONTH"] = label_encoder.fit_transform(data["MONTH"])

    # Split features and target variable
    X = data.drop('Wind ', axis=1,errors='ignore')
    y = data['Wind ']

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
    #fres.append(mse_ensemble)
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
    fres.append(rmse_ensemble)
    #fres.append(mape_ensemble)
    fres.append(r2_ensemble)

    # Group the data by year and calculate the mean Wind   power generation for each year
    wind_yearly_mean = data.groupby('YEAR')['Wind '].mean().reset_index()

    # Plot the trend of Wind  power generation over the years
    plt.figure(figsize=(10, 6))
    plt.plot(wind_yearly_mean['YEAR'], wind_yearly_mean['Wind '], marker='o', linestyle='-')
    plt.title('Trend of Wind  Power Generation Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Wind  Power Generation')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig("static/Wind.jpg")
    #plt.show()
