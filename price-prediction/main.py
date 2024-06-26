import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#ucitavanje podataka iz csv fajla (podaci sa linka: https://www.kaggle.com/datasets/muhammetvarl/laptop-price?resource=download)
#1304 redova
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

#Podaci će biti podvrgnuti standardnim postupcima pretprocesiranja kao što su uklanjanje duplikata, 
#tretiranje nedostajućih vrednosti, normalizacija ili standardizacija numeričkih atributa,
#enkodiranje kategoričkih atributa, kao i podela skupa podataka na trening, validacioni i testni skup.
def preprocess_data(data):
    
    #x-> skup podataka sa atributima bez cene 
    #y-> podaci o ceni odnosno ciljnom obeležju koje želimo predvideti
    X = data.drop(['Price'], axis=1)
    y = data['Price']
    
    #uklanjamo sufiks kg iz atributa Tezina (Weight) kako bismo ga mogli koristiti u numerickom obliku
    X['Weight'] = X['Weight'].str.replace('kg', '').astype(float)
    
    #definisemo numericke i kategoricke atribute
    numeric_features = ['Inches', 'Weight']
    categorical_features = ['Company', 'TypeName', 'Cpu', 'Memory', 'Gpu', 'OpSys', 'Ram']
    
    #definisemo transformacije koje cemo primeniti na podatke pre nego ih iskoristimo za obuku modela
    #SimpleImputer- pipeline za numericke atribute, koristi se za popunjavanje nedostajućih vrednosti
    #zatim se koristi StandardScaler radi standardizacije podataka
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    #pipeline za kategoricke atribute- SimpleImputer (za popunjavanje nedostajucih vrednosti)
    #zatim se primenjuje OneHotEncoder koji kategoricke promenljive pretvara u binarne dummy podatke
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Use most frequent value for missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    #ColumnTransformer- objedinjuje transformacije (numericke i kategoricke)
    #u jednu celinu koja moze da se primeni na sve atribute podataka
    #sta je ColumnTransformer uopste??
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    #primenjujemo sve te transformacije na podatke X--> koje sve tacno semo uklanjanja nedostajucih vrednosti??
    #i kako primenjujemo?? sta je ovo fit_transform
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor  #vracamo transformisane podatke, ciljni atribut y i objekat za transformaciju (koji se kasnije moze koristiti za dalje transformisanje podataka)

def preprocess_weight(weight_str):
    if weight_str.strip() == '':
        return 2.0  # Default weight if nothing is entered
    else:
        return float(weight_str.replace('kg', '').strip())  # Remove 'kg' and convert to float

def preprocess_new_data(new_data, preprocessor):
    # Process weight separately
    new_data['Weight'] = new_data['Weight'].apply(preprocess_weight)
    
    # Preprocess using preprocessor from training data
    new_data_processed = preprocessor.transform(new_data)
    
    return new_data_processed

def main():
    # Load data
    file_path = r'C:\Users\Administrator\Desktop\price-prediction\laptop_data.csv'  # Replace with your actual file path
    data = load_data(file_path)
    
    # Preprocess data
    X_processed, y, preprocessor = preprocess_data(data)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Example: Make predictions for new data
    print("Enter laptop details:")
    new_data = {}
    new_data['Company'] = input("Company: ").strip() or 'Lenovo'
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
    
    # Convert new_data to DataFrame
    new_data = pd.DataFrame(new_data, index=[0])
    
    # Preprocess new data (similar to training data)
    new_data_processed = preprocess_new_data(new_data, preprocessor)
    
    # Predict price for new data
    predicted_price = model.predict(new_data_processed)
    
    print(f'Predicted Price: {predicted_price}')

if __name__ == "__main__":
    main()
