import pandas as pd
import numpy as np
# data loading and cleaning functions
# parse functions for specific columns in the dataset we use
def parse_price(val):
    if isinstance(val, str):
        val = val.replace('TL', '').replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return np.nan
    return val

def parse_area(val):
    if isinstance(val, str):
        val = val.replace(' m2', '').replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return np.nan
    return val
# oda sayilari 2+1 3+1 gibi ifadeleri handle et, topla direkt mesela 3+1 4 olacak gibi
# turkiyede ve datasette genelde +1 oluyor
def parse_rooms(val):
    if isinstance(val, str):
        val = val.replace(' Oda', '')
        if '+' in val:
            try:
                parts = val.split('+')
                return float(parts[0]) + float(parts[1])
            except:
                return np.nan
        try:
            return float(val)
        except:
            return np.nan
    return val

def parse_age(val):
    if isinstance(val, str):
        if 'Ve Üzeri' in val:
            return 25.0 # Approximation
        if 'Sıfır' in val:
            return 0.0
        if '-' in val:
            try:
                parts = val.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
    return val
#datayi loadla
def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    #relevant columns that we are going to use in this project
    cols_to_keep = [
        'district', 'NumberOfRooms', 'GrossSquareMeters', 'NetSquareMeters', 
        'BuildingAge', 'FloorLocation', 'HeatingType', 'ItemStatus', 'price'
    ]
    
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]
    
    print("Cleaning data...")
    df['price'] = df['price'].apply(parse_price)

    #parse functionlari calistir
    if 'GrossSquareMeters' in df.columns:
        df['GrossSquareMeters'] = df['GrossSquareMeters'].apply(parse_area)
    if 'NetSquareMeters' in df.columns:
        df['NetSquareMeters'] = df['NetSquareMeters'].apply(parse_area)
        
    if 'NumberOfRooms' in df.columns:
        df['NumberOfRooms'] = df['NumberOfRooms'].apply(parse_rooms)
        
    if 'BuildingAge' in df.columns:
        df['BuildingAge'] = df['BuildingAge'].apply(parse_age)
        df['BuildingAge'] = pd.to_numeric(df['BuildingAge'], errors='coerce')
    # price olmayan rowlari kaldir
    df = df.dropna(subset=['price'])
    
    return df

def load_and_clean_data(filepath):
    df = load_and_parse_data(filepath)
    
    # quantile range ile extreme outlierlari kaldir
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR)))]
    
    print(f"Data shape after cleaning: {df.shape}")
    return df

def load_and_parse_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    #relevant columns that we are going to use in this project
    cols_to_keep = [
        'district', 'NumberOfRooms', 'GrossSquareMeters', 'NetSquareMeters', 
        'BuildingAge', 'FloorLocation', 'HeatingType', 'ItemStatus', 'price'
    ]
    
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]
    
    print("Parsing data...")
    df['price'] = df['price'].apply(parse_price)

    #parse functionlari calistir
    if 'GrossSquareMeters' in df.columns:
        df['GrossSquareMeters'] = df['GrossSquareMeters'].apply(parse_area)
    if 'NetSquareMeters' in df.columns:
        df['NetSquareMeters'] = df['NetSquareMeters'].apply(parse_area)
        
    if 'NumberOfRooms' in df.columns:
        df['NumberOfRooms'] = df['NumberOfRooms'].apply(parse_rooms)
        
    if 'BuildingAge' in df.columns:
        df['BuildingAge'] = df['BuildingAge'].apply(parse_age)
        df['BuildingAge'] = pd.to_numeric(df['BuildingAge'], errors='coerce')
    # price olmayan rowlari kaldir
    df = df.dropna(subset=['price'])
    return df

# train test split 80 20 

def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X.iloc[train_idx].copy(), X.iloc[test_idx].copy(), y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

def get_polynomial_features(X_num):
    #featureleri degree 2 yapma fonksiyonu
    X = np.array(X_num)
    n_samples, n_features = X.shape
    
    #asil featureleri ekle
    poly_features = [X]
    
    #ikili carpimlari ekle ve return et
    for i in range(n_features):
        for j in range(i, n_features):
            new_col = X[:, i] * X[:, j]
            poly_features.append(new_col.reshape(-1, 1))
            
    return np.hstack(poly_features)
# feature isimlerini polynomial featurelara gore olustur
def get_poly_feature_names(numeric_cols):
    names = list(numeric_cols)
    n = len(names)
    for i in range(n):
        for j in range(i, n):
            names.append(f"{names[i]}*{names[j]}")
    return names

def prepare_data(df):
    target = 'price'
    X = df.drop(columns=[target])
    y = df[target]
    
    # 1. Train Test Split (Manual)
    print("Splitting data (Manual)...")
    X_train_raw, X_test_raw, y_train, y_test = manual_train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    #categorical featurelerin string oldugunu kontrol et
    for col in categorical_features:
        X_train_raw[col] = X_train_raw[col].astype(str)
        X_test_raw[col] = X_test_raw[col].astype(str)
    
    print(f"Numerical features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # numerical featureleri isle
    print("Processing numerical features (Imputation -> Poly -> Scaling)...")
    
    # Imputation
    train_medians = X_train_raw[numeric_features].median()
    X_train_num = X_train_raw[numeric_features].fillna(train_medians).values
    X_test_num = X_test_raw[numeric_features].fillna(train_medians).values
    
    # feature al
    X_train_poly = get_polynomial_features(X_train_num)
    X_test_poly = get_polynomial_features(X_test_num)
    
    poly_names = get_poly_feature_names(numeric_features)
    
    # standardization part just like in the lecture notes
    mean = np.mean(X_train_poly, axis=0)
    std = np.std(X_train_poly, axis=0)
    std[std == 0] = 1.0 #division by zero prevention
    
    X_train_scaled = (X_train_poly - mean) / std
    X_test_scaled = (X_test_poly - mean) / std
    
    # categorical featureleri isle
    print("Processing categorical features (Imputation -> OneHot)...")
    
    # Imputation
    X_train_cat = X_train_raw[categorical_features].fillna('missing')
    X_test_cat = X_test_raw[categorical_features].fillna('missing')
    
    # one hot encoding
    X_train_encoded_list = []
    X_test_encoded_list = []
    cat_names = []
    
    for col in categorical_features:
        # Train setten unique kategorileri al
        unique_cats = sorted(X_train_cat[col].unique())
        
        for cat in unique_cats:
            col_name = f"{col}_{cat}"
            cat_names.append(col_name)
            
            # Train
            X_train_encoded_list.append((X_train_cat[col] == cat).astype(float).values.reshape(-1, 1))
            
            # Test
            X_test_encoded_list.append((X_test_cat[col] == cat).astype(float).values.reshape(-1, 1))
            
    X_train_cat_encoded = np.hstack(X_train_encoded_list)
    X_test_cat_encoded = np.hstack(X_test_encoded_list)
    
    # ikisini stackle 
    X_train_processed = np.hstack([X_train_scaled, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_scaled, X_test_cat_encoded])
    
    feature_names = poly_names + cat_names
    
    return X_train_processed, X_test_processed, y_train.values, y_test.values, feature_names

if __name__ == "__main__":
    from pathlib import Path
    project_dir = Path(__file__).resolve().parent
    df = load_and_clean_data(project_dir / 'HouseData.csv')
    X_train, X_test, y_train, y_test, features = prepare_data(df)
    
    #saving function
    np.savez(project_dir / 'processed_data.npz', 
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=features)
    print("Data preparation complete. Saved to processed_data.npz")
