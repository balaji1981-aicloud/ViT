import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

def prepare_data(data_23, data_24, data_25, material_data):
    df = pd.concat([data_23, data_24, data_25], ignore_index=True)
    
    df = df[
        (df['Billing type Name'] == 'Invoice') & 
        (df['Company code'] != 3025) & 
        (df['Interco/External'] == 'R04')
    ]
    
    df['Material'] = df['Material'].astype('Int64')
    df = df.merge(material_data, left_on='Material', right_on='MaterialNum')
    
    df = df[
        (df['Customer (Bill to) Name'] != 'NATIONAL RESEARCH CO.') &
        (df['R05'] != 0) &
        (df['R05'].notna())
    ]
    
    return df


def aggregate_by_period(df):
    agg_df = df.groupby(
        ['Customer (Bill to) Name', 'MaterialGroupDesc', 'Fiscal year/period'], 
        as_index=False
    ).agg({
        'Billing Quantity SU': ['sum', 'count'],
        'R05': 'sum',
        'Gross Margin': 'sum'
    })
    
    agg_df.columns = ['Customer', 'Product', 'Period', 'Quantity', 'Transactions', 'Selling_Price', 'Margin']
    
    agg_df['period'] = pd.to_datetime(agg_df['Period'], format="%b %Y")
    agg_df['year'] = agg_df['period'].dt.year
    
    return agg_df


def filter_active_customers(df):
    order_counts = df.groupby(['Customer', 'Product']).size()
    multi_order_mask = df.set_index(['Customer', 'Product']).index.map(order_counts) > 1
    df = df[multi_order_mask].copy()
    
    yearly_presence = df.groupby(['Customer', 'Product', 'year']).size().unstack(fill_value=0)
    active_both_years = yearly_presence[
        (yearly_presence.get(2023, 0) > 0) & 
        (yearly_presence.get(2024, 0) > 0)
    ]
    
    active_mask = df.set_index(['Customer', 'Product']).index.isin(active_both_years.index)
    return df[active_mask].copy()


def calculate_trend_analysis(df):
    def slope_calc(series):
        if len(series) < 2:
            return np.nan
        slope, _, _, _, _ = linregress(range(len(series)), series)
        return slope
    
    yearly = df.groupby(['Customer', 'Product', 'year']).agg({
        'Transactions': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    trends = yearly.groupby(['Customer', 'Product']).agg({
        'Transactions': slope_calc,
        'Quantity': slope_calc
    }).reset_index()
    
    trends.columns = ['Customer', 'Product', 'frequency_trend', 'quantity_trend']
    return trends


def prepare_for_modeling(df):
    trans = df[['Customer', 'Product', 'period', 'Quantity']].copy()
    trans['Customer_Product'] = trans['Customer'] + "_" + trans['Product']
    
    summary = summary_data_from_transaction_data(
        trans, 
        customer_id_col='Customer_Product', 
        datetime_col='period', 
        freq='D', 
        freq_multiplier=30
    )
    
    avg_qty = trans.groupby('Customer_Product')['Quantity'].mean()
    summary = summary.join(avg_qty.rename('quantity_value'))
    summary = summary[summary['quantity_value'] > 0].copy()
    
    return summary, trans


def validate_and_clean_inputs(frequency, recency, T, quantity=None):
    freq = np.array(frequency)
    rec = np.array(recency)
    t = np.array(T)
    
    freq = np.where(freq >= 0, freq, 0)
    rec = np.clip(rec, 0, t)
    t = np.where(t > 0, t, 1)
    
    if quantity is not None:
        qty = np.array(quantity)
        valid_qty = qty[qty > 0]
        median_qty = np.median(valid_qty) if len(valid_qty) > 0 else 1
        qty = np.where(qty > 0, qty, median_qty)
        return freq, rec, t, qty
    
    return freq, rec, t


def fit_clv_models(summary, penalizer=0.01, max_retries=3):
    freq, rec, t, qty = validate_and_clean_inputs(
        summary['frequency'], 
        summary['recency'], 
        summary['T'],
        summary['quantity_value']
    )
    
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    
    for attempt in range(max_retries):
        try:
            bgf.fit(freq, rec, t, iterative_fitting=attempt+1, verbose=False)
            if hasattr(bgf, 'summary') and bgf.summary is not None:
                break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Warning: BG/NBD fitting failed after {max_retries} attempts")
            freq = freq + np.random.normal(0, 0.01, len(freq))
    
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    
    returning_mask = (freq > 0) & (qty > 0)
    fit_freq = freq[returning_mask]
    fit_qty = qty[returning_mask]
    
    if len(fit_freq) > 0:
        for attempt in range(max_retries):
            try:
                ggf.fit(fit_freq, fit_qty, iterative_fitting=attempt+1, verbose=False)
                if hasattr(ggf, 'summary') and ggf.summary is not None:
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Warning: Gamma-Gamma fitting failed after {max_retries} attempts")
                fit_qty = fit_qty + np.random.normal(0, np.std(fit_qty)*0.01, len(fit_qty))
    
    return bgf, ggf


def generate_predictions(summary, bgf, ggf, horizon_days=90, batch_size=1000):
    results = summary.copy()
    n_rows = len(results)
    
    predicted_purchases = np.zeros(n_rows)
    predicted_avg_value = np.zeros(n_rows)
    
    for start_idx in range(0, n_rows, batch_size):
        end_idx = min(start_idx + batch_size, n_rows)
        batch = results.iloc[start_idx:end_idx]
        
        b_freq, b_rec, b_t, b_qty = validate_and_clean_inputs(
            batch['frequency'],
            batch['recency'],
            batch['T'],
            batch['quantity_value']
        )
        
        try:
            purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
                horizon_days, b_freq, b_rec, b_t
            )
            predicted_purchases[start_idx:end_idx] = np.where(
                np.isfinite(purchases), 
                purchases, 
                b_freq / b_t * horizon_days
            )
        except:
            predicted_purchases[start_idx:end_idx] = b_freq / b_t * horizon_days
        
        try:
            avg_value = ggf.conditional_expected_average_profit(b_freq, b_qty)
            predicted_avg_value[start_idx:end_idx] = np.where(
                np.isfinite(avg_value), 
                avg_value, 
                b_qty
            )
        except:
            predicted_avg_value[start_idx:end_idx] = b_qty
    
    smoothing = 0.95
    results['exp_avg_quantity'] = (
        predicted_avg_value * smoothing + 
        results['quantity_value'] * (1 - smoothing)
    )
    results['exp_purchases_next90d'] = np.maximum(0, predicted_purchases)
    results['expected_total_quantity_next90d'] = (
        results['exp_avg_quantity'] * results['exp_purchases_next90d']
    )
    
    results['data_quality_score'] = np.clip(
        results['frequency'] / results['frequency'].quantile(0.75),
        0, 1
    )
    results['prediction_confidence'] = 0.85 + results['data_quality_score'] * 0.15
    results['adjusted_prediction'] = (
        results['expected_total_quantity_next90d'] * 
        results['prediction_confidence']
    )
    
    return results


df = prepare_data(data_23, data_24, data_25, data_dach)
df_agg = aggregate_by_period(df)
df_filtered = filter_active_customers(df_agg)

trends = calculate_trend_analysis(df_filtered)

train_summary, train_trans = prepare_for_modeling(df_filtered)

bgf_model, ggf_model = fit_clv_models(train_summary, penalizer=0.01)

final_predictions = generate_predictions(
    train_summary, 
    bgf_model, 
    ggf_model, 
    horizon_days=90,
    batch_size=1000
)

df_validation = df_filtered[~df_filtered['Period'].str.contains('2025')].copy()
val_summary, val_trans = prepare_for_modeling(df_validation)

validation_bgf, validation_ggf = fit_clv_models(val_summary, penalizer=0.001)

validation_predictions = generate_predictions(
    val_summary,
    validation_bgf,
    validation_ggf,
    horizon_days=90,
    batch_size=1000
)