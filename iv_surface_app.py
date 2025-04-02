# surface_iv_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf  # Importing yfinance to fetch stock prices
from iv_surface import debug_iv_surface

def fetch_current_price(ticker):
    """Fetch the current price of the given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.history(period="1d")['Close'].iloc[0]  # Get the latest closing price

def main():
    st.set_page_config(page_title="IV Surface Explorer", layout="wide")

    # Page Title
    st.title("Implied Volatility Surface")

    # Sidebar for parameters
    with st.sidebar:
        # Model Parameters
        st.header("Model Parameters")
        # Here the user inputs decimals for the rates
        # e.g. 0.015 => 1.5% (later multiplied by 100)
        risk_free_rate_decimal = 0.015
        dividend_yield_decimal = 0.013
            

        # Visualization Parameters
        st.header("Visualization Parameters")
        y_axis_mode = st.selectbox(
            "Select Y-axis:",
            ("Strike Price ($)", "Moneyness (K (Strike Price) / S (Current Price))")
        )
        show_points = st.checkbox("Show Actual Points", value=False)
        show_surface = st.checkbox("Show Surface Plot", value=True)

        # Ticker Input
        st.header("Ticker Symbol")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()

        # Generate Button
        generate_button = st.button("Generate IV Surface")

    # Main content area
    if generate_button:
        try:
            with st.spinner("Generating Implied Volatility Surface..."):
                # Fetch current price for the ticker symbol
                current_price = fetch_current_price(ticker)

                # Display current price at the top
                st.subheader(f"Current Price of {ticker}: ${current_price:.2f}")

                # Convert decimals to percentages for iv_surface
                r_as_percentage = risk_free_rate_decimal * 100.0
                div_as_percentage = dividend_yield_decimal * 100.0

                # Call debug_iv_surface
                options_df, fig = debug_iv_surface(
                    ticker_symbol=ticker,
                    max_days=730,
                    r=r_as_percentage,
                    div_yield=div_as_percentage,
                    y_axis=y_axis_mode,
                    show_points=True
                )

                # Handle the three cases
                if show_surface and show_points:
                    
                    # Case 2: Show both surface and points
                    fig.update_layout(coloraxis_colorbar=dict(title='Implied Volatility (%)'))
                    
                    # Add points using the correct y-values based on the selected mode
                    if y_axis_mode == "Strike Price ($)":
                        y_points = options_df['strike'].values
                    else:
                        y_points = options_df['strike'].values / current_price  # Calculate moneyness

                    fig.add_trace(go.Scatter3d(
                        x=options_df['T'].values,
                        y=y_points,
                        z=options_df['impliedVol'].values,
                        mode='markers',
                        marker=dict(size=8, color='red', opacity=0.7),
                        name='Data Points'
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                elif show_surface:
                    # Case 3: Show only surface
                    fig.update_layout(coloraxis_colorbar=dict(title='Implied Volatility (%)'))
                    st.plotly_chart(fig, use_container_width=True)

                elif show_points:
                    # Case 1: Show only points
                    fig.data = []  # Clear existing data
                    if y_axis_mode == "Strike Price ($)":
                        y_points = options_df['strike'].values
                    else:
                        y_points = options_df['strike'].values / current_price  # Calculate moneyness

                    fig.add_trace(go.Scatter3d(
                        x=options_df['T'].values,
                        y=y_points,
                        z=options_df['impliedVol'].values,
                        mode='markers',
                        marker=dict(size=8, color='red', opacity=0.7),
                        name='Data Points'
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                # Display Raw Data
                with st.expander("View Raw Data"):
                    st.dataframe(options_df)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
