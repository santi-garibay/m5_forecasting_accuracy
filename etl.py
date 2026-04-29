import pandas as pd
import numpy as np
import gc
import os

# Configuration
DATA_DIR = 'data/'
OUTPUT_FILE = 'm5_analysis.parquet' 
METRICS_FILE = 'memory_metrics.csv'

def reduce_mem_usage(df):
    """
    Optimiza los tipos de datos iterando columnas.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    # 1. Optimizar Objetos a Categorías
    id_vars = df.select_dtypes(include=['object']).columns
    for col in id_vars:
        df[col] = df[col].astype('category')

    # 2. Downcasting de Números
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'   -> Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def run_etl():
    print("[PHASE 1] Loading & Melting Sales Data...")
    df_sales = pd.read_csv(os.path.join(DATA_DIR, 'sales_train_validation.csv'))
    
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(df_sales, id_vars=id_vars, var_name='d', value_name='sales')
    
    del df_sales
    gc.collect()

    print("[PHASE 2] Initial Memory Optimization...")
    df = reduce_mem_usage(df)
    
    print("[PHASE 3] Merging Calendar...")
    calendar = pd.read_csv(os.path.join(DATA_DIR, 'calendar.csv'))
    calendar = calendar[['d', 'date', 'wm_yr_wk', 'weekday', 'event_name_1', 'snap_CA', 'snap_TX', 'snap_WI']]
    df = pd.merge(df, calendar, on='d', how='left')
    del calendar
    df = reduce_mem_usage(df) 
    
    print("[PHASE 4] Merging Prices...")
    prices = pd.read_csv(os.path.join(DATA_DIR, 'sell_prices.csv'))
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    del prices
    gc.collect()
    df = reduce_mem_usage(df) 

    print("[PHASE 5] Feature Engineering...")
    df['revenue'] = df['sales'] * df['sell_price']
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # SNAP Logic
    print("   -> Calculating SNAP active days...")
    conditions = [
        (df['state_id'] == 'CA') & (df['snap_CA'] == 1),
        (df['state_id'] == 'TX') & (df['snap_TX'] == 1),
        (df['state_id'] == 'WI') & (df['snap_WI'] == 1)
    ]
    df['snap_active'] = np.select(conditions, [True, True, True], default=False)
    df.drop(columns=['snap_CA', 'snap_TX', 'snap_WI', 'wm_yr_wk'], inplace=True)

    print("[PHASE 6] Calculating Metrics & Saving...")
    
    # --- CÁLCULO REALISTA AJUSTADO ---
    
    # 1. Tamaño Real Actual (Optimizado)
    final_mem_size = df.memory_usage(deep=True).sum() / 1024**2
    
    # 2. Estimación "Si no hubieras optimizado" (Standard Pandas)
    num_rows = len(df)
    
    # Identificamos columnas que originalmente eran texto (Strings pesan ~70-80 bytes en Pandas)
    # ids, item, dept, cat, store, state, d, weekday, event_name, date (si es str)
    cols_objects = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd', 'weekday', 'event_name_1', 'date']
    num_obj_cols = len([c for c in cols_objects if c in df.columns])
    
    # Columnas numéricas (8 bytes standard float64/int64)
    # sales, sell_price, year, month, day, revenue, snap_active
    num_numeric_cols = len(df.columns) - num_obj_cols
    
    # Fórmula: (Filas * ColsNum * 8 bytes) + (Filas * ColsObj * 80 bytes)
    # 80 bytes es un promedio conservador para strings en Python (overhead + data)
    size_numeric = num_rows * num_numeric_cols * 8
    size_objects = num_rows * num_obj_cols * 80 
    
    estimated_unoptimized_size = (size_numeric + size_objects) / 1024**2
    
    metrics = pd.DataFrame({
        'Metric': ['Standard Pandas (Est.)', 'Optimized Pipeline'],
        'Size_MB': [estimated_unoptimized_size, final_mem_size]
    })
    
    metrics.to_csv(METRICS_FILE, index=False)
    print(f"   -> Metrics Saved.")
    print(f"   -> Theoretical Unoptimized: {estimated_unoptimized_size:.0f} MB")
    print(f"   -> Actual Optimized: {final_mem_size:.0f} MB")

    if OUTPUT_FILE.endswith('.csv'):
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        df.to_parquet(OUTPUT_FILE, index=False)
        
    print("ETL Completed Successfully.")

if __name__ == "__main__":
    run_etl()