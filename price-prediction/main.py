import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    return data

def preprocess_data(data):
    data.drop_duplicates(inplace=True)

    X = data.drop(['Price_euros'], axis=1)
    y = data['Price_euros']
    
    X['Weight'] = X['Weight'].str.replace('kg', '').astype(float)
  
    numeric_features = ['Inches', 'Weight']
    categorical_features = ['Company', 'Product', 'TypeName', 'Cpu', 'Memory', 'Gpu', 'OpSys', 'Ram']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor  

def preprocess_weight(weight_str):
    if weight_str.strip() == '':
        return 2.0 
    else:
        return float(weight_str.replace('kg', '').strip()) 

def preprocess_new_data(new_data, preprocessor):
    new_data['Weight'] = new_data['Weight'].apply(preprocess_weight)
    new_data_processed = preprocessor.transform(new_data)
    return new_data_processed

def main():
    file_path = 'laptop_price0.csv'
    data = load_data(file_path)
    
    # Preprocess data
    X_processed, y, preprocessor = preprocess_data(data)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(random_state=42),
        'Ridge Regression': Ridge()
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        print(f'Cross-validation MAE: {-cv_scores_mae.mean()}')
        print(f'Cross-validation R2-score: {cv_scores_r2.mean()}')

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluation on the test set
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Test MAE: {mae}')
        print(f'Test R2-score: {r2}')
    
    # User input for new data
    print("Enter laptop details:")
    new_data = {}
    new_data['Company'] = input("Company: ").strip() or 'Lenovo'
    new_data['Product'] = input("Product: ").strip() or 'IdeaPad 320-15ISK'
    new_data['TypeName'] = input("Type Name: ").strip() or 'Notebook'
    inches_input = input("Inches: ").strip() or '15.6'
    new_data['Inches'] = float(inches_input)
    new_data['ScreenResolution'] = input("Screen Resolution: ").strip() or 'Full HD'
    new_data['Cpu'] = input("CPU: ").strip() or 'Intel Core i5 2.4GHz'
    new_data['Ram'] = input("RAM: ").strip() or '8GB'
    new_data['Memory'] = input("Memory: ").strip() or '256GB SSD'
    new_data['Gpu'] = input("GPU: ").strip() or 'NVIDIA GeForce GTX 1650'
    new_data['OpSys'] = input("Operating System: ").strip() or 'Windows 10'
    weight_input = input("Weight (in kg): ").strip() or '2.0'
    new_data['Weight'] = weight_input

    # Convert user input to DataFrame
    new_data = pd.DataFrame(new_data, index=[0])

    # Preprocess new data
    new_data_processed = preprocess_new_data(new_data, preprocessor)

    # Predict price for new data using all models
    for model_name, model in models.items():
        predicted_price = model.predict(new_data_processed)
        print(f'Predicted Price ({model_name}): {predicted_price}')

if __name__ == "__main__":
    main()
