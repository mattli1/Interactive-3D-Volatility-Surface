# iv_surface.py

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import mibian
from scipy.interpolate import griddata

def fetch_options_data(
    ticker_symbol: str,
    max_expiries=8,
    moneyness_range=(0.7, 1.3),
    max_days=730,
    progress_callback=None
):
    """
    Fetch options data for up to 'max_days' in the future.
    'moneyness_range' is in decimal form, e.g. (0.7, 1.3) means
    70%-130% of the spot price.
    """
    try:
        if progress_callback:
            progress_callback("Fetching ticker data...", 10)

        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='1d')
        if hist.empty:
            raise ValueError(f"No price history found for {ticker_symbol}.")

        current_price = hist['Close'].iloc[-1]

        # All expiration strings from yfinance
        all_expirations = ticker.options
        today = datetime.datetime.now()

        # Filter valid expirations within max_days
        valid_expirations = []
        for exp in all_expirations:
            try:
                exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d")
                days_diff = (exp_date - today).days
                if 0 <= days_diff <= max_days:
                    valid_expirations.append(exp)
            except:
                pass

        # Down-select to at most 'max_expiries' dates
        max_expiries = min(max_expiries, len(valid_expirations))
        if len(valid_expirations) > max_expiries:
            indices = np.linspace(0, len(valid_expirations) - 1, max_expiries, dtype=int)
            expirations = [valid_expirations[i] for i in indices]
        else:
            expirations = valid_expirations

        if progress_callback:
            progress_callback(f"Processing {len(expirations)} expiration dates", 20)

        all_options_data = []
        total_exps = len(expirations)

        # Fetch call options for each expiration
        for i, exp_date in enumerate(expirations):
            progress = 30 + (40 * i / total_exps)
            if progress_callback:
                progress_callback(f"Processing expiration {i+1}/{total_exps}", progress)

            try:
                chain = ticker.option_chain(exp_date)

                # Focus on calls for illustration
                df = chain.calls[['strike', 'bid', 'ask', 'lastPrice']].copy()
                df['expirationDate'] = exp_date
                df['optionType'] = 'c'

                # Filter strikes by moneyness
                mask = (
                    (df['strike'] >= current_price * moneyness_range[0]) &
                    (df['strike'] <= current_price * moneyness_range[1])
                )
                filtered_df = df[mask]

                # Sample ~15 data points to keep data size manageable
                if len(filtered_df) > 15:
                    sample_indices = np.linspace(
                        0, len(filtered_df) - 1, 15, dtype=int
                    )
                    filtered_df = filtered_df.iloc[sample_indices]

                all_options_data.append(filtered_df)

            except Exception as e:
                print(f"Warning: Error processing {exp_date}: {e}")
                continue

        options_df = pd.concat(all_options_data, ignore_index=True)

        if progress_callback:
            progress_callback(f"Processing {len(options_df)} total contracts", 80)

        return options_df, current_price

    except Exception as e:
        raise Exception(f"Error fetching options data: {str(e)}")

def compute_time_to_expiration(expiration_date_str: str) -> float:
    """
    Convert a YYYY-MM-DD expiration string to days until expiration.
    Returns 0 if expiration is in the past (no negative values).
    """
    try:
        expiration_date = datetime.datetime.strptime(expiration_date_str, "%Y-%m-%d")
        today = datetime.datetime.now()
        days_to_exp = (expiration_date - today).days
        return max(days_to_exp, 0)
    except Exception as e:
        raise Exception(f"Error computing time to expiration: {str(e)}")

def compute_iv(row, underlying_price, r=1.5, div_yield=1.3):
    """
    Compute implied volatility using Mibian.
    'r' and 'div_yield' are in percentage form, e.g. 1.5 => 1.5%.
    Mibian interprets them literally as 1.5% (not 0.015).
    """
    try:
        K = float(row['strike'])
        market_price = float(row['lastPrice'])
        if pd.isna(market_price) or market_price <= 0:
            return np.nan

        # Time in years
        T = compute_time_to_expiration(row['expirationDate']) / 365.0
        if T <= 0:
            return np.nan

        # Mibian uses [underPrice, strikePrice, interestRate, daysToExp]
        days_to_exp = int(T * 365)
        c = mibian.BS([underlying_price, K, r, days_to_exp], callPrice=market_price)
        iv = c.impliedVolatility  # e.g. 30 => 30%

        # Basic sanity check
        if iv > 500 or iv < 1:
            return np.nan

        return iv
    except Exception:
        return np.nan

def debug_iv_surface(
    ticker_symbol: str,
    moneyness_range=(0.7, 1.3),
    max_days=730,
    verbose=True,
    progress_callback=None,
    r=1.5,           # 1.5 => 1.5%
    div_yield=1.3,   # 1.3 => 1.3%
    y_axis="Strike Price ($)",
    show_points=False
):
    """
    Create a 3D implied volatility surface using Plotly, with normalization
    on the (x, y) axes before using griddata (cubic interpolation).

    Steps:
      1) Fetch + filter data
      2) Compute T and IV
      3) Decide y-axis: strike or moneyness
      4) Normalize x_data (T) and y_data before cubic interpolation
      5) Build the final surface with un-normalized mesh grid (so visually it shows original scales)
    """
    try:
        # 1) Fetch data
        options_df, current_price = fetch_options_data(
            ticker_symbol,
            max_days=max_days,
            moneyness_range=moneyness_range,
            progress_callback=progress_callback
        )

        if options_df.empty:
            raise ValueError("No options data available.")

        # 2) Compute T in years + IV
        options_df['T'] = options_df['expirationDate'].apply(compute_time_to_expiration) / 365.0
        options_df['impliedVol'] = options_df.apply(
            lambda row: compute_iv(row, current_price, r=r, div_yield=div_yield),
            axis=1
        )
        options_df.dropna(subset=['impliedVol'], inplace=True)
        options_df['impliedVol'] = options_df['impliedVol'].clip(lower=0, upper=100)

        if options_df.empty:
            raise ValueError("Insufficient valid data to generate surface.")

        # 3) Decide y-values
        if y_axis == "Strike Price ($)":
            y_vals = options_df['strike'].values
        else:
            y_vals = (options_df['strike'] / current_price).values

        x_vals = options_df['T'].values
        z_vals = options_df['impliedVol'].values

        # 4) Normalize x_data, y_data
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()

        # Avoid division by zero if there's no variation
        if x_max == x_min:
            x_max = x_min + 1e-6
        if y_max == y_min:
            y_max = y_min + 1e-6

        x_norm = (x_vals - x_min) / (x_max - x_min)
        y_norm = (y_vals - y_min) / (y_max - y_min)

        # 5) Build a normalized mesh grid from 0..1
        XN, YN = np.meshgrid(
            np.linspace(0, 1, 100),
            np.linspace(0, 1, 100)
        )

        # Interpolate with griddata on normalized coords
        z_grid_norm = griddata(
            (x_norm, y_norm),
            z_vals,
            (XN, YN),
            method='cubic'
        )

        # Fill NaNs with the min IV, then clip
        iv_min = z_vals.min()
        z_grid_norm = np.where(np.isnan(z_grid_norm), iv_min, z_grid_norm)
        z_grid_norm = np.clip(z_grid_norm, 0, 100)

        # Un-normalize XN, YN back to real scales for plotting
        X = XN * (x_max - x_min) + x_min
        Y = YN * (y_max - y_min) + y_min

        # 6) Build Plotly figure
        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=z_grid_norm,
            colorscale='Plasma',
            cmin=0,
            cmax=100
        )])

        # Lock z-axis to [0..100]
        fig.update_layout(
            title=f'Implied Volatility Surface for {ticker_symbol}',
            scene=dict(
                xaxis_title='Time to Expiration (Years)',
                yaxis_title=y_axis,
                zaxis=dict(range=[0, 100], title='Implied Volatility (%)')
            ),
            width=1200,
            height=800
        )

        return options_df, fig

    except Exception as e:
        raise Exception(f"Error generating IV surface: {str(e)}")
