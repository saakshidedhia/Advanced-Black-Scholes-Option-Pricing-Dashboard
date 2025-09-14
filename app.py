import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from BlackScholes import AdvancedBlackScholes
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Black-Scholes Option Pricing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        color: #e0e0e0;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #2a2a2a, #3a3a3a);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 1px solid #4a4a4a;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
    }
    
    .metric-title {
        font-size: 14px;
        color: #888;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00ff88;
        margin-bottom: 5px;
    }
    
    .metric-call {
        border-left: 4px solid #00ff88;
    }
    
    .metric-put {
        border-left: 4px solid #ff6b6b;
    }
    
    .greek-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0;
    }
    
    .greek-item {
        background: #2a2a2a;
        padding: 15px;
        border-radius: 10px;
        flex: 1;
        min-width: 120px;
        text-align: center;
        border: 1px solid #4a4a4a;
    }
    
    .greek-name {
        font-size: 12px;
        color: #888;
        margin-bottom: 5px;
    }
    
    .greek-value {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4a4a4a, #5a5a5a);
        color: #ffffff;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #5a5a5a, #6a6a6a);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    .sidebar .stNumberInput > div > div > input {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border: 1px solid #4a4a4a;
        border-radius: 5px;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #2a2a2a 0%, #3a3a3a 100%);
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    .analysis-section {
        background: #2a2a2a;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #4a4a4a;
    }
</style>
""", unsafe_allow_html=True)

def create_price_surface():
    """Create 3D surface plot for option pricing."""
    spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 20)
    vol_range = np.linspace(0.1, 0.5, 20)
    
    call_surface = np.zeros((len(vol_range), len(spot_range)))
    put_surface = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            temp_bs = AdvancedBlackScholes(
                spot_price=spot,
                strike_price=strike,
                time_to_expiry=time_to_maturity,
                volatility=vol,
                risk_free_rate=interest_rate,
                dividend_yield=dividend_yield
            )
            prices = temp_bs.calculate_prices()
            call_surface[i, j] = prices['call']
            put_surface[i, j] = prices['put']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call Option Surface', 'Put Option Surface'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    fig.add_trace(
        go.Surface(
            x=spot_range,
            y=vol_range,
            z=call_surface,
            colorscale='Viridis',
            name='Call',
            showscale=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Surface(
            x=spot_range,
            y=vol_range,
            z=put_surface,
            colorscale='Plasma',
            name='Put'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Option Price Surfaces",
        height=600,
        scene=dict(
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            zaxis_title="Option Price",
            bgcolor="rgba(0,0,0,0)"
        ),
        scene2=dict(
            xaxis_title="Spot Price",
            yaxis_title="Volatility",
            zaxis_title="Option Price",
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_greeks_radar_chart(greeks_data):
    """Create radar chart for Greeks visualization."""
    call_greeks = greeks_data['call']
    put_greeks = greeks_data['put']
    
    # Normalize Greeks for radar chart (scale appropriately)
    greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    call_values = [
        call_greeks['delta'],
        call_greeks['gamma'] * 100,  # Scale gamma
        abs(call_greeks['theta']) * 10,  # Scale theta and make positive
        call_greeks['vega'],
        call_greeks['rho']
    ]
    put_values = [
        abs(put_greeks['delta']),  # Make positive for comparison
        put_greeks['gamma'] * 100,
        abs(put_greeks['theta']) * 10,
        put_greeks['vega'],
        abs(put_greeks['rho'])
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=call_values,
        theta=greek_names,
        fill='toself',
        name='Call',
        line_color='#00ff88'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=put_values,
        theta=greek_names,
        fill='toself',
        name='Put',
        line_color='#ff6b6b'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(call_values), max(put_values)) * 1.1]
            )),
        showlegend=True,
        title="Greeks Comparison (Scaled for Visualization)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_sensitivity_analysis():
    """Create sensitivity analysis charts."""
    base_price = current_price
    price_range = np.linspace(base_price * 0.8, base_price * 1.2, 50)
    
    call_prices = []
    put_prices = []
    call_deltas = []
    put_deltas = []
    
    for price in price_range:
        temp_bs = AdvancedBlackScholes(
            spot_price=price,
            strike_price=strike,
            time_to_expiry=time_to_maturity,
            volatility=volatility,
            risk_free_rate=interest_rate,
            dividend_yield=dividend_yield
        )
        prices = temp_bs.calculate_prices()
        greeks = temp_bs.calculate_greeks()
        
        call_prices.append(prices['call'])
        put_prices.append(prices['put'])
        call_deltas.append(greeks['call']['delta'])
        put_deltas.append(greeks['put']['delta'])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Option Prices vs Spot Price', 'Delta vs Spot Price', 
                       'Price Difference', 'Payoff Diagram'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Option prices
    fig.add_trace(
        go.Scatter(x=price_range, y=call_prices, name='Call Price', 
                  line=dict(color='#00ff88', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=price_range, y=put_prices, name='Put Price', 
                  line=dict(color='#ff6b6b', width=3)),
        row=1, col=1
    )
    
    # Delta
    fig.add_trace(
        go.Scatter(x=price_range, y=call_deltas, name='Call Delta', 
                  line=dict(color='#00ff88', dash='dash')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=price_range, y=put_deltas, name='Put Delta', 
                  line=dict(color='#ff6b6b', dash='dash')),
        row=1, col=2
    )
    
    # Price difference
    price_diff = np.array(call_prices) - np.array(put_prices)
    fig.add_trace(
        go.Scatter(x=price_range, y=price_diff, name='Call - Put', 
                  line=dict(color='#ffff00', width=2)),
        row=2, col=1
    )
    
    # Payoff diagrams
    call_payoff = np.maximum(price_range - strike, 0)
    put_payoff = np.maximum(strike - price_range, 0)
    
    fig.add_trace(
        go.Scatter(x=price_range, y=call_payoff, name='Call Payoff', 
                  line=dict(color='#00ff88', dash='dot', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=price_range, y=put_payoff, name='Put Payoff', 
                  line=dict(color='#ff6b6b', dash='dot', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

# Sidebar for inputs
with st.sidebar:
    st.title("🎯 Option Parameters")
    
    st.markdown("### Market Data")
    current_price = st.number_input("Current Spot Price ($)", value=100.0, min_value=0.01, step=0.01)
    strike = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=0.01)
    time_to_maturity = st.number_input("Time to Expiry (Years)", value=0.25, min_value=0.001, step=0.001)
    
    st.markdown("### Risk Parameters")
    volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.20, step=0.01, format="%.2f")
    interest_rate = st.slider("Risk-Free Rate", min_value=-0.05, max_value=0.20, value=0.05, step=0.001, format="%.3f")
    dividend_yield = st.slider("Dividend Yield", min_value=0.0, max_value=0.10, value=0.0, step=0.001, format="%.3f")
    
    st.markdown("---")
    st.markdown("### Analysis Options")
    show_greeks = st.checkbox("Show Greeks Analysis", value=True)
    show_sensitivity = st.checkbox("Show Sensitivity Analysis", value=True)
    show_3d_surface = st.checkbox("Show 3D Price Surface", value=False)

# Main content
st.title("🚀 Advanced Black-Scholes Option Pricing Dashboard")

# Create Black-Scholes model
try:
    bs_model = AdvancedBlackScholes(
        spot_price=current_price,
        strike_price=strike,
        time_to_expiry=time_to_maturity,
        volatility=volatility,
        risk_free_rate=interest_rate,
        dividend_yield=dividend_yield
    )
    
    prices = bs_model.calculate_prices()
    greeks = bs_model.calculate_greeks()
    summary = bs_model.get_summary()
    
    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-call">
            <div class="metric-title">Call Option</div>
            <div class="metric-value">${prices['call']:.4f}</div>
            <div style="font-size: 12px; color: #888;">
                Intrinsic: ${summary['intrinsic_value']['call']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-put">
            <div class="metric-title">Put Option</div>
            <div class="metric-value">${prices['put']:.4f}</div>
            <div style="font-size: 12px; color: #888;">
                Intrinsic: ${summary['intrinsic_value']['put']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        moneyness_color = "#00ff88" if summary['moneyness'] > 1 else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Moneyness</div>
            <div class="metric-value" style="color: {moneyness_color};">{summary['moneyness']:.4f}</div>
            <div style="font-size: 12px; color: #888;">
                S/K Ratio
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        days_to_expiry = time_to_maturity * 365
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Time Value</div>
            <div class="metric-value">${summary['time_value']['call']:.4f}</div>
            <div style="font-size: 12px; color: #888;">
                {days_to_expiry:.0f} days remaining
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Parameters table
    st.markdown("### 📋 Input Parameters")
    
    input_data = {
        "Parameter": ["Spot Price", "Strike Price", "Time to Expiry", "Volatility", "Risk-Free Rate", "Dividend Yield"],
        "Value": [f"${current_price:.2f}", f"${strike:.2f}", f"{time_to_maturity:.4f} years", 
                 f"{volatility:.2%}", f"{interest_rate:.3%}", f"{dividend_yield:.3%}"],
        "Description": ["Current asset price", "Exercise price", "Time until expiration", 
                       "Annual volatility", "Risk-free interest rate", "Continuous dividend yield"]
    }
    
    df = pd.DataFrame(input_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Greeks display
    if show_greeks:
        st.markdown("### 🔢 Greeks Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Call Option Greeks")
            call_greeks_html = f"""
            <div class="greek-container">
                <div class="greek-item">
                    <div class="greek-name">Delta</div>
                    <div class="greek-value">{greeks['call']['delta']:.4f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Gamma</div>
                    <div class="greek-value">{greeks['call']['gamma']:.6f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Theta</div>
                    <div class="greek-value">{greeks['call']['theta']:.4f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Vega</div>
                    <div class="greek-value">{greeks['call']['vega']:.4f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Rho</div>
                    <div class="greek-value">{greeks['call']['rho']:.4f}</div>
                </div>
            </div>
            """
            st.markdown(call_greeks_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Put Option Greeks")
            put_greeks_html = f"""
            <div class="greek-container">
                <div class="greek-item">
                    <div class="greek-name">Delta</div>
                    <div class="greek-value">{greeks['put']['delta']:.4f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Gamma</div>
                    <div class="greek-value">{greeks['put']['gamma']:.6f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Theta</div>
                    <div class="greek-value">{greeks['put']['theta']:.4f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Vega</div>
                    <div class="greek-value">{greeks['put']['vega']:.4f}</div>
                </div>
                <div class="greek-item">
                    <div class="greek-name">Rho</div>
                    <div class="greek-value">{greeks['put']['rho']:.4f}</div>
                </div>
            </div>
            """
            st.markdown(put_greeks_html, unsafe_allow_html=True)
        
        # Greeks explanation
        with st.expander("📚 Greeks Explanation"):
            st.markdown("""
            **Delta**: Price sensitivity to underlying asset price changes (≈ hedge ratio)
            - Call delta: 0 to 1, Put delta: -1 to 0
            
            **Gamma**: Rate of change of delta (convexity)
            - Higher gamma means delta changes more rapidly
            
            **Theta**: Time decay (usually negative for long positions)
            - How much option price decreases per day
            
            **Vega**: Sensitivity to volatility changes
            - How much option price changes for 1% volatility change
            
            **Rho**: Sensitivity to interest rate changes
            - How much option price changes for 1% interest rate change
            """)
        
        # Radar chart for Greeks
        st.plotly_chart(create_greeks_radar_chart(greeks), use_container_width=True)
    
    # Sensitivity Analysis
    if show_sensitivity:
        st.markdown("### 📈 Sensitivity Analysis")
        sensitivity_fig = create_sensitivity_analysis()
        st.plotly_chart(sensitivity_fig, use_container_width=True)
    
    # 3D Surface Plot
    if show_3d_surface:
        st.markdown("### 🌊 3D Option Price Surface")
        with st.spinner("Generating 3D surface plots..."):
            surface_fig = create_price_surface()
            st.plotly_chart(surface_fig, use_container_width=True)
    
    # Implied Volatility Calculator
    st.markdown("### 🎯 Implied Volatility Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_price = st.number_input("Market Option Price", value=prices['call'], min_value=0.01, step=0.01)
    
    with col2:
        option_type = st.selectbox("Option Type", ["call", "put"])
    
    with col3:
        if st.button("Calculate IV"):
            with st.spinner("Calculating implied volatility..."):
                iv = bs_model.implied_volatility(market_price, option_type)
                if iv is not None:
                    st.success(f"Implied Volatility: {iv:.4f} ({iv:.2%})")
                else:
                    st.error("Failed to converge. Try a different market price.")
    
    # Risk Management Section
    st.markdown("### ⚠️ Risk Management Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-section">
            <h4>Position Risk Assessment</h4>
        """, unsafe_allow_html=True)
        
        # Risk metrics
        max_loss_call = prices['call']
        max_loss_put = prices['put']
        breakeven_call = strike + prices['call']
        breakeven_put = strike - prices['put']
        
        st.write(f"**Call Option:**")
        st.write(f"• Max Loss: ${max_loss_call:.2f} (premium paid)")
        st.write(f"• Breakeven: ${breakeven_call:.2f}")
        st.write(f"• Probability of Profit: {(1 - norm.cdf((np.log(breakeven_call/current_price) + (interest_rate - 0.5*volatility**2)*time_to_maturity)/(volatility*np.sqrt(time_to_maturity))))*100:.1f}%")
        
        st.write(f"\n**Put Option:**")
        st.write(f"• Max Loss: ${max_loss_put:.2f} (premium paid)")
        st.write(f"• Breakeven: ${breakeven_put:.2f}")
        st.write(f"• Probability of Profit: {norm.cdf((np.log(breakeven_put/current_price) + (interest_rate - 0.5*volatility**2)*time_to_maturity)/(volatility*np.sqrt(time_to_maturity)))*100:.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analysis-section">
            <h4>Market Outlook</h4>
        """, unsafe_allow_html=True)
        
        if summary['moneyness'] > 1.05:
            st.write("📈 **Bullish Signal**: Current price significantly above strike")
        elif summary['moneyness'] < 0.95:
            st.write("📉 **Bearish Signal**: Current price significantly below strike")
        else:
            st.write("⚖️ **Neutral**: Price near strike price")
        
        if volatility > 0.3:
            st.write("🌪️ **High Volatility**: Consider selling strategies")
        elif volatility < 0.15:
            st.write("😴 **Low Volatility**: Consider buying strategies")
        else:
            st.write("📊 **Normal Volatility**: Standard market conditions")
        
        if time_to_maturity < 0.08:  # Less than 30 days
            st.write("⏰ **Short Time**: High time decay risk")
        elif time_to_maturity > 0.5:  # More than 6 months
            st.write("📅 **Long Time**: Lower time decay impact")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Monte Carlo Simulation Section
    st.markdown("### 🎲 Monte Carlo Price Simulation")
    
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running simulation..."):
            # Monte Carlo parameters
            n_simulations = 10000
            n_steps = max(1, int(time_to_maturity * 252))  # Daily steps
            dt = time_to_maturity / n_steps
            
            # Generate price paths
            np.random.seed(42)  # For reproducibility
            price_paths = np.zeros((n_simulations, n_steps + 1))
            price_paths[:, 0] = current_price
            
            for i in range(n_steps):
                z = np.random.standard_normal(n_simulations)
                price_paths[:, i + 1] = price_paths[:, i] * np.exp(
                    (interest_rate - dividend_yield - 0.5 * volatility**2) * dt + 
                    volatility * np.sqrt(dt) * z
                )
            
            # Calculate payoffs
            final_prices = price_paths[:, -1]
            call_payoffs = np.maximum(final_prices - strike, 0)
            put_payoffs = np.maximum(strike - final_prices, 0)
            
            # Discount to present value
            call_mc_price = np.exp(-interest_rate * time_to_maturity) * np.mean(call_payoffs)
            put_mc_price = np.exp(-interest_rate * time_to_maturity) * np.mean(put_payoffs)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Monte Carlo Call Price", f"${call_mc_price:.4f}", 
                         f"{call_mc_price - prices['call']:+.4f}")
            
            with col2:
                st.metric("Monte Carlo Put Price", f"${put_mc_price:.4f}", 
                         f"{put_mc_price - prices['put']:+.4f}")
            
            with col3:
                st.metric("Price Range", f"${final_prices.min():.2f} - ${final_prices.max():.2f}")
            
            # Plot price distribution
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=final_prices,
                nbinsx=50,
                name='Final Prices',
                marker_color='rgba(0, 255, 136, 0.7)'
            ))
            
            fig_hist.add_vline(x=strike, line_dash="dash", line_color="red", 
                              annotation_text="Strike Price")
            fig_hist.add_vline(x=current_price, line_dash="dash", line_color="yellow", 
                              annotation_text="Current Price")
            
            fig_hist.update_layout(
                title="Simulated Final Price Distribution",
                xaxis_title="Stock Price",
                yaxis_title="Frequency",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)

except Exception as e:
    st.error(f"Error in calculations: {str(e)}")
    st.info("Please check your input parameters and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; margin-top: 20px;'>
    <p>⚠️ This tool is for educational purposes only. Not financial advice.</p>
    <p>Built with ❤️ using Streamlit | Enhanced Black-Scholes Model v2.0</p>
</div>
""", unsafe_allow_html=True)