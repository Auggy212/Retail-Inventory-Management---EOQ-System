"""
📦 AI-Driven Retail Inventory Optimization using the EOQ Model
A Streamlit app for inventory management and optimization

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EOQ Inventory Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_eoq(annual_demand, ordering_cost, holding_cost):
    """
    Calculate Economic Order Quantity (EOQ)
    
    Formula: EOQ = sqrt((2 * D * S) / H)
    Where:
        D = Annual Demand
        S = Ordering Cost per Order
        H = Holding Cost per Unit per Year
    """
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return eoq

def calculate_total_cost(annual_demand, eoq, ordering_cost, holding_cost):
    """
    Calculate total annual inventory cost
    
    TC = (D/EOQ) * S + (EOQ/2) * H
    """
    ordering_cost_total = (annual_demand / eoq) * ordering_cost
    holding_cost_total = (eoq / 2) * holding_cost
    total_cost = ordering_cost_total + holding_cost_total
    
    return total_cost, ordering_cost_total, holding_cost_total

def simulate_inventory(annual_demand, eoq, reorder_point, demand_variation=0):
    """
    Simulate monthly inventory levels for 12 months
    
    Args:
        annual_demand: Total demand for the year
        eoq: Economic Order Quantity
        reorder_point: Inventory level that triggers reorder
        demand_variation: Percentage variation in monthly demand (0-100)
    
    Returns:
        DataFrame with monthly simulation data
    """
    monthly_demand_base = annual_demand / 12
    inventory = eoq  # Start with full EOQ
    orders_placed = 1  # Initial order
    
    simulation_data = []
    
    for month in range(1, 13):
        # Add random variation to demand if specified
        if demand_variation > 0:
            variation = np.random.uniform(-demand_variation/100, demand_variation/100)
            monthly_demand = max(0, monthly_demand_base * (1 + variation))
        else:
            monthly_demand = monthly_demand_base
        
        monthly_demand = round(monthly_demand)
        
        # Check if reorder is needed BEFORE consuming demand
        received_order = 0
        if inventory <= reorder_point:
            received_order = eoq
            inventory += eoq
            orders_placed += 1
        
        # Consume demand
        inventory = max(0, inventory - monthly_demand)
        
        # Store month data
        simulation_data.append({
            'Month': f'M{month}',
            'Month_Num': month,
            'Demand': monthly_demand,
            'Order_Received': received_order,
            'Ending_Inventory': round(inventory),
            'Below_Reorder': inventory < reorder_point
        })
    
    df = pd.DataFrame(simulation_data)
    return df, orders_placed

def create_inventory_chart(df, reorder_point, eoq):
    """
    Create interactive Plotly chart for inventory visualization
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Inventory Levels Over Time', 'Monthly Demand Pattern'),
        vertical_spacing=0.15,
        row_heights=[0.65, 0.35]
    )
    
    # Main inventory line chart
    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=df['Ending_Inventory'],
            mode='lines+markers+text',
            name='Ending Inventory',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10, color='#3b82f6'),
            text=df['Ending_Inventory'],
            textposition='top center',
            textfont=dict(size=10, color='#1e40af'),
            hovertemplate='<b>%{x}</b><br>Inventory: %{y} units<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add reorder point reference line
    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=[reorder_point] * len(df),
            mode='lines',
            name='Reorder Point',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='Reorder Point: %{y} units<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add EOQ reference line
    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=[eoq] * len(df),
            mode='lines',
            name='EOQ Level',
            line=dict(color='#10b981', width=2, dash='dot'),
            hovertemplate='EOQ: %{y} units<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Highlight reorder events
    reorder_months = df[df['Order_Received'] > 0]['Month']
    if len(reorder_months) > 0:
        fig.add_trace(
            go.Scatter(
                x=reorder_months,
                y=df[df['Order_Received'] > 0]['Ending_Inventory'],
                mode='markers',
                name='Order Placed',
                marker=dict(size=15, color='#a855f7', symbol='star'),
                hovertemplate='<b>Order Placed</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Demand bar chart
    fig.add_trace(
        go.Bar(
            x=df['Month'],
            y=df['Demand'],
            name='Monthly Demand',
            marker=dict(color='#10b981'),
            hovertemplate='<b>%{x}</b><br>Demand: %{y} units<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Inventory (Units)", row=1, col=1)
    fig.update_yaxes(title_text="Demand (Units)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_cost_breakdown_chart(ordering_cost, holding_cost):
    """
    Create pie chart for cost breakdown
    """
    fig = go.Figure(data=[go.Pie(
        labels=['Ordering Cost', 'Holding Cost'],
        values=[ordering_cost, holding_cost],
        hole=0.4,
        marker=dict(colors=['#8b5cf6', '#ec4899']),
        textinfo='label+percent+value',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Annual Cost Breakdown",
        height=400,
        showlegend=True
    )
    
    return fig

def generate_insights(eoq, orders_placed, total_cost, avg_inventory, df, theoretical_orders):
    """
    Generate actionable insights based on simulation results
    """
    insights = []
    
    # EOQ Efficiency
    insights.append({
        'type': 'success',
        'title': '✅ EOQ Optimization',
        'text': f'The Economic Order Quantity of {eoq:.0f} units balances ordering and holding costs effectively. '
                f'This results in {orders_placed} orders per year (theoretical: {theoretical_orders:.1f}).'
    })
    
    # Inventory Management
    stockout_months = len(df[df['Ending_Inventory'] == 0])
    if stockout_months > 0:
        insights.append({
            'type': 'warning',
            'title': '⚠️ Stockout Alert',
            'text': f'Stockouts occurred in {stockout_months} month(s). Consider increasing safety stock or adjusting reorder point.'
        })
    else:
        insights.append({
            'type': 'success',
            'title': '✅ No Stockouts',
            'text': 'Excellent! No stockouts occurred during the simulation period. The reorder point is well-calibrated.'
        })
    
    # Cost Efficiency
    insights.append({
        'type': 'info',
        'title': '💰 Cost Analysis',
        'text': f'Total annual inventory cost is ${total_cost:,.2f}. Average inventory maintained at {avg_inventory:.0f} units, '
                f'representing approximately {(avg_inventory/eoq)*100:.1f}% of EOQ.'
    })
    
    # Reorder Pattern
    reorder_frequency = len(df[df['Order_Received'] > 0])
    insights.append({
        'type': 'info',
        'title': '📊 Ordering Pattern',
        'text': f'Orders were placed {reorder_frequency} times during the year. '
                f'Average time between orders: {12/reorder_frequency:.1f} months.'
    })
    
    return insights

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📦 Retail Inventory Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Driven Economic Order Quantity (EOQ) Model</div>', unsafe_allow_html=True)
    
    # Sidebar - Input Parameters
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 Demand & Cost Parameters")
    annual_demand = st.sidebar.number_input(
        "Annual Demand (units)",
        min_value=100,
        max_value=1000000,
        value=12000,
        step=100,
        help="Total number of units needed per year"
    )
    
    ordering_cost = st.sidebar.number_input(
        "Ordering Cost ($ per order)",
        min_value=1.0,
        max_value=10000.0,
        value=50.0,
        step=5.0,
        help="Fixed cost incurred each time an order is placed"
    )
    
    holding_cost = st.sidebar.number_input(
        "Holding Cost ($ per unit per year)",
        min_value=0.1,
        max_value=1000.0,
        value=2.0,
        step=0.5,
        help="Cost to store one unit for one year"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Simulation Settings")
    
    demand_variation = st.sidebar.slider(
        "Demand Variation (±%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Random variation in monthly demand"
    )
    
    reorder_percentage = st.sidebar.slider(
        "Reorder Point (% of EOQ)",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
        help="Inventory level that triggers a new order"
    )
    
    run_monte_carlo = st.sidebar.checkbox(
        "🎲 Run Monte Carlo Simulation",
        value=False,
        help="Run 1000 simulations to analyze uncertainty"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip**: The EOQ model helps minimize total inventory costs by finding the optimal order quantity "
        "that balances ordering costs and holding costs."
    )
    
    # Calculate EOQ
    eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost)
    reorder_point = (reorder_percentage / 100) * eoq
    
    # Calculate costs
    total_cost, ordering_cost_total, holding_cost_total = calculate_total_cost(
        annual_demand, eoq, ordering_cost, holding_cost
    )
    
    theoretical_orders = annual_demand / eoq
    
    # Run simulation
    df_simulation, orders_placed = simulate_inventory(
        annual_demand, eoq, reorder_point, demand_variation
    )
    
    avg_inventory = df_simulation['Ending_Inventory'].mean()
    
    # ========================================================================
    # KEY METRICS DISPLAY
    # ========================================================================
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 EOQ Value",
            value=f"{eoq:.0f} units",
            delta=f"{eoq/annual_demand*100:.1f}% of annual demand"
        )
    
    with col2:
        st.metric(
            label="💰 Total Annual Cost",
            value=f"${total_cost:,.2f}",
            delta=f"{orders_placed} orders placed"
        )
    
    with col3:
        st.metric(
            label="📈 Orders Per Year",
            value=f"{orders_placed}",
            delta=f"Theoretical: {theoretical_orders:.1f}"
        )
    
    with col4:
        st.metric(
            label="📊 Avg Inventory",
            value=f"{avg_inventory:.0f} units",
            delta=f"{(avg_inventory/eoq)*100:.1f}% of EOQ"
        )
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    st.markdown("---")
    st.markdown("## 📈 Inventory Visualization")
    
    # Create and display main chart
    fig_inventory = create_inventory_chart(df_simulation, reorder_point, eoq)
    st.plotly_chart(fig_inventory, use_container_width=True)
    
    # ========================================================================
    # DETAILED ANALYSIS
    # ========================================================================
    st.markdown("---")
    st.markdown("## 🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost breakdown
        fig_cost = create_cost_breakdown_chart(ordering_cost_total, holding_cost_total)
        st.plotly_chart(fig_cost, use_container_width=True)
        
        st.markdown("### 💵 Cost Details")
        cost_df = pd.DataFrame({
            'Cost Component': ['Ordering Cost', 'Holding Cost', 'Total Cost'],
            'Amount ($)': [
                f"${ordering_cost_total:,.2f}",
                f"${holding_cost_total:,.2f}",
                f"${total_cost:,.2f}"
            ],
            'Percentage': [
                f"{(ordering_cost_total/total_cost)*100:.1f}%",
                f"{(holding_cost_total/total_cost)*100:.1f}%",
                "100.0%"
            ]
        })
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
    
    with col2:
        # EOQ Formula Display
        st.markdown("### 📐 EOQ Formula")
        st.latex(r"EOQ = \sqrt{\frac{2DS}{H}}")
        st.markdown(f"""
        Where:
        - **D** (Annual Demand) = {annual_demand:,} units
        - **S** (Ordering Cost) = ${ordering_cost:.2f}
        - **H** (Holding Cost) = ${holding_cost:.2f}
        
        **Calculated EOQ** = {eoq:.0f} units
        """)
        
        st.markdown("### 📊 Order Economics")
        st.markdown(f"""
        - **Order Frequency**: Every {12/theoretical_orders:.1f} months
        - **Reorder Point**: {reorder_point:.0f} units
        - **Safety Stock**: {reorder_point:.0f} units ({reorder_percentage}% of EOQ)
        - **Max Inventory**: {eoq:.0f} units
        """)
    
    # ========================================================================
    # SIMULATION DATA TABLE
    # ========================================================================
    st.markdown("---")
    st.markdown("## 📋 Monthly Simulation Data")
    
    # Format the dataframe for display
    display_df = df_simulation.copy()
    display_df['Order_Received'] = display_df['Order_Received'].apply(
        lambda x: f"✅ {x} units" if x > 0 else "—"
    )
    display_df['Status'] = display_df['Below_Reorder'].apply(
        lambda x: "⚠️ Below Reorder" if x else "✅ Normal"
    )
    display_df = display_df[['Month', 'Demand', 'Order_Received', 'Ending_Inventory', 'Status']]
    display_df.columns = ['Month', 'Demand (units)', 'Order Received', 'Ending Inventory (units)', 'Status']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button for simulation data
    csv = df_simulation.to_csv(index=False)
    st.download_button(
        label="📥 Download Simulation Data (CSV)",
        data=csv,
        file_name=f"inventory_simulation_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # ========================================================================
    # MONTE CARLO SIMULATION (OPTIONAL)
    # ========================================================================
    if run_monte_carlo:
        st.markdown("---")
        st.markdown("## 🎲 Monte Carlo Simulation Analysis")
        st.info("Running 1,000 simulations with varying demand patterns...")
        
        # Run multiple simulations
        n_simulations = 1000
        results = {
            'total_costs': [],
            'orders_placed': [],
            'avg_inventories': [],
            'stockouts': []
        }
        
        progress_bar = st.progress(0)
        for i in range(n_simulations):
            df_sim, orders = simulate_inventory(annual_demand, eoq, reorder_point, demand_variation)
            sim_cost, _, _ = calculate_total_cost(annual_demand, eoq, ordering_cost, holding_cost)
            
            results['total_costs'].append(sim_cost)
            results['orders_placed'].append(orders)
            results['avg_inventories'].append(df_sim['Ending_Inventory'].mean())
            results['stockouts'].append(len(df_sim[df_sim['Ending_Inventory'] == 0]))
            
            if i % 10 == 0:
                progress_bar.progress((i + 1) / n_simulations)
        
        progress_bar.empty()
        
        # Display Monte Carlo results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💰 Avg Total Cost",
                f"${np.mean(results['total_costs']):,.2f}",
                delta=f"±${np.std(results['total_costs']):.2f}"
            )
        
        with col2:
            st.metric(
                "📦 Avg Orders",
                f"{np.mean(results['orders_placed']):.1f}",
                delta=f"±{np.std(results['orders_placed']):.1f}"
            )
        
        with col3:
            stockout_prob = (sum([1 for x in results['stockouts'] if x > 0]) / n_simulations) * 100
            st.metric(
                "⚠️ Stockout Probability",
                f"{stockout_prob:.1f}%",
                delta=f"In {n_simulations} simulations"
            )
        
        # Distribution plots
        fig_monte = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Cost Distribution', 'Orders Placed Distribution')
        )
        
        fig_monte.add_trace(
            go.Histogram(x=results['total_costs'], name='Total Cost', nbinsx=30,
                        marker=dict(color='#3b82f6')),
            row=1, col=1
        )
        
        fig_monte.add_trace(
            go.Histogram(x=results['orders_placed'], name='Orders', nbinsx=10,
                        marker=dict(color='#10b981')),
            row=1, col=2
        )
        
        fig_monte.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_monte, use_container_width=True)
    
    # ========================================================================
    # INSIGHTS & RECOMMENDATIONS
    # ========================================================================
    st.markdown("---")
    st.markdown("## 🧠 AI-Generated Insights & Recommendations")
    
    insights = generate_insights(eoq, orders_placed, total_cost, avg_inventory, df_simulation, theoretical_orders)
    
    for insight in insights:
        if insight['type'] == 'success':
            st.markdown(f"""
            <div class="success-box">
                <strong>{insight['title']}</strong><br>
                {insight['text']}
            </div>
            """, unsafe_allow_html=True)
        elif insight['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-box">
                <strong>{insight['title']}</strong><br>
                {insight['text']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
                <strong>{insight['title']}</strong><br>
                {insight['text']}
            </div>
            """, unsafe_allow_html=True)
    
    # Additional recommendations
    st.markdown("### 💡 Optimization Recommendations")
    st.markdown(f"""
    1. **Review Ordering Costs**: Current ordering cost is ${ordering_cost:.2f}. Negotiate bulk discounts with suppliers to reduce this.
    2. **Monitor Demand Patterns**: Current variation is ±{demand_variation}%. Implement demand forecasting to reduce uncertainty.
    3. **Safety Stock Analysis**: Consider increasing reorder point if stockouts are unacceptable for your business.
    4. **Supplier Lead Time**: Adjust reorder point based on actual supplier lead times to prevent stockouts.
    5. **Seasonal Adjustments**: If demand is seasonal, consider dynamic EOQ calculations per quarter.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem;'>
        <p>📦 <strong>EOQ Inventory Optimizer</strong> | Built with Streamlit & Python</p>
        <p>🎯 Optimize your inventory management with data-driven insights</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
