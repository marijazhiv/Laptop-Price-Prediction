import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor

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
        ]
    )
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

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Cross-validation
    cv_scores_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    print(f'\n{model_name} - Cross-validation MAE: {-cv_scores_mae.mean()}')
    print(f'{model_name} - Cross-validation R2-score: {cv_scores_r2.mean()}')

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluation on the test set
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{model_name} - Test MAE: {mae}')
    print(f'{model_name} - Test R2-score: {r2}')

    return model

def main():
    file_path = 'laptop_price0.csv'
    data = load_data(file_path)

    # Preprocess data
    X_processed, y, preprocessor = preprocess_data(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Initialize models without hyperparameter tuning
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'Linear Regression': LinearRegression()
    }

    # Train and evaluate models without hyperparameter tuning
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name} without hyperparameter tuning...")
        model.fit(X_train, y_train)
        models[model_name] = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)

    # Initialize hyperparameter grids
    models_and_grids = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'search': RandomizedSearchCV,
            'search_params': {
                'n_iter': 10,
                'verbose': 1,
                'n_jobs': -1
            }
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'param_grid': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'search': RandomizedSearchCV,
            'search_params': {
                'n_iter': 10,
                'verbose': 1,
                'n_jobs': -1
            }
        },
        'Extra Trees': {
            'model': ExtraTreesRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'search': RandomizedSearchCV,
            'search_params': {
                'n_iter': 10,
                'verbose': 1,
                'n_jobs': -1
            }
        },
        'Ridge Regression': {
            'model': Ridge(),
            'param_grid': {
                'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000]
            },
            'search': GridSearchCV,
            'search_params': {
                'verbose': 1
            }
        },
        'Linear Regression': {
            'model': LinearRegression(),
            'param_grid': {},  # No hyperparameters to tune
            'search': None,
            'search_params': {}
        }
    }

    best_models = {}

    # Train and tune hyperparameters for each model
    for model_name, config in models_and_grids.items():
        model = config['model']
        param_grid = config['param_grid']
        search = config['search']
        search_params = config['search_params']

        if search:
            print(f"\nTuning hyperparameters for {model_name}...")
            if model_name == 'Ridge Regression':
                grid_search = search(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', **search_params)
            else:
                grid_search = search(estimator=model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', **search_params)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model

            best_params = grid_search.best_params_
            print(f'Best parameters for {model_name}: {best_params}')

            # Evaluate optimized models
            best_models[model_name] = evaluate_model(best_model, X_train, y_train, X_test, y_test, f'{model_name} (optimized)')
        else:
            # Linear Regression model doesn't require tuning
            best_models[model_name] = model
            best_models[model_name].fit(X_train, y_train)
            evaluate_model(best_models[model_name], X_train, y_train, X_test, y_test, f'{model_name} (non-optimized)')

    # Initialize best Ridge model separately since it's used later
    best_ridge = best_models['Ridge Regression']

    while True:
        # User input for new data
        print("\nEnter laptop details (or type 'exit' to stop):")
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

        if new_data['Company'].lower() == 'exit':
            break

        # Convert user input to DataFrame
        new_data = pd.DataFrame(new_data, index=[0])

        # Preprocess new data
        new_data_processed = preprocess_new_data(new_data, preprocessor)

        # Predict price for new data using all models (non-optimized)
        print("\nPredicted prices with non-optimized models:")
        for model_name, model in models.items():
            predicted_price = model.predict(new_data_processed)
            print(f'Predicted Price ({model_name}): {predicted_price}')

        # Predict price for new data using all best models (optimized)
        print("\nPredicted prices with optimized models:")
        for model_name, model in best_models.items():
            predicted_price_best = model.predict(new_data_processed)
            print(f'Predicted Price ({model_name}): {predicted_price_best}')

if __name__ == "__main__":
    main()

