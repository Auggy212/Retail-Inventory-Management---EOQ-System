import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Inventory Optimization Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .highlight {
        background-color: #ffeaa7;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

class InventoryOptimizer:
    """EOQ-based Inventory Optimization Engine"""
    
    def __init__(self, annual_demand, ordering_cost, holding_cost, reorder_percentage=0.25):
        self.annual_demand = annual_demand
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.reorder_percentage = reorder_percentage
        
        # Calculate EOQ
        self.eoq = self._calculate_eoq()
        self.reorder_point = self.eoq * reorder_percentage
        self.num_orders = self.annual_demand / self.eoq
        self.total_annual_cost = self._calculate_total_cost()
        
    def _calculate_eoq(self):
        """Calculate Economic Order Quantity"""
        eoq = np.sqrt((2 * self.annual_demand * self.ordering_cost) / self.holding_cost)
        return int(round(eoq))
    
    def _calculate_total_cost(self):
        """Calculate total annual inventory cost"""
        ordering_cost = (self.annual_demand / self.eoq) * self.ordering_cost
        holding_cost = (self.eoq / 2) * self.holding_cost
        return ordering_cost + holding_cost
    
    def simulate_inventory(self, months=12, demand_variation=0.1):
        """Simulate monthly inventory levels"""
        inventory = self.eoq
        results = []
        total_orders = 0
        
        base_monthly_demand = self.annual_demand / 12
        
        for month in range(1, months + 1):
            # Generate monthly demand with variation
            variation = np.random.uniform(-demand_variation, demand_variation)
            monthly_demand = int(base_monthly_demand * (1 + variation))
            
            # Check if reorder needed
            received_order = 0
            order_placed = False
            if inventory <= self.reorder_point:
                received_order = self.eoq
                inventory += received_order
                total_orders += 1
                order_placed = True
            
            # Deduct demand
            starting_inventory = inventory
            inventory = max(0, inventory - monthly_demand)
            
            # Store results
            results.append({
                'Month': month,
                'Starting_Inventory': starting_inventory,
                'Demand': monthly_demand,
                'Order_Placed': order_placed,
                'Order_Quantity': received_order,
                'Ending_Inventory': inventory,
                'Stockout': inventory == 0
            })
        
        return pd.DataFrame(results), total_orders

def create_inventory_plot(df, eoq, reorder_point):
    """Create interactive Plotly visualization"""
    fig = go.Figure()
    
    # Main inventory line
    fig.add_trace(go.Scatter(
        x=df['Month'],
        y=df['Ending_Inventory'],
        mode='lines+markers+text',
        name='Inventory Level',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=10),
        text=df['Ending_Inventory'].astype(str),
        textposition="top center",
        textfont=dict(size=12, color='black', family='Arial Black'),
        hovertemplate='Month: %{x}<br>Inventory: %{y} units<extra></extra>'
    ))
    
    # Highlight reorder points
    reorder_df = df[df['Order_Placed']]
    if not reorder_df.empty:
        fig.add_trace(go.Scatter(
            x=reorder_df['Month'],
            y=reorder_df['Ending_Inventory'],
            mode='markers',
            name='Reorder Points',
            marker=dict(
                size=15,
                color='#E63946',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='Reorder placed<br>New stock: %{y} units<extra></extra>'
        ))
    
    # Add reorder level line
    fig.add_hline(
        y=reorder_point,
        line_dash="dash",
        line_color="#E63946",
        annotation_text=f"Reorder Point ({int(reorder_point)} units)",
        annotation_position="right"
    )
    
    # Add EOQ level line
    fig.add_hline(
        y=eoq,
        line_dash="dot",
        line_color="#06D6A0",
        annotation_text=f"EOQ Level ({eoq} units)",
        annotation_position="left"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "📊 Inventory Depletion & Replenishment using EOQ Model",
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        xaxis_title="Month",
        yaxis_title="Inventory Level (Units)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxis(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        dtick=1
    )
    fig.update_yaxis(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def create_cost_breakdown_chart(optimizer, total_orders):
    """Create cost breakdown pie chart"""
    ordering_cost = total_orders * optimizer.ordering_cost
    avg_inventory = optimizer.eoq / 2
    holding_cost = avg_inventory * optimizer.holding_cost
    
    fig = go.Figure(data=[go.Pie(
        labels=['Ordering Cost', 'Holding Cost'],
        values=[ordering_cost, holding_cost],
        hole=0.3,
        marker_colors=['#2E86AB', '#06D6A0']
    )])
    
    fig.update_layout(
        title="Annual Cost Breakdown",
        showlegend=True,
        height=300
    )
    
    return fig

def generate_insights(optimizer, df, total_orders):
    """Generate insights and recommendations"""
    avg_inventory = df['Ending_Inventory'].mean()
    stockout_months = len(df[df['Stockout']])
    demand_cv = df['Demand'].std() / df['Demand'].mean()
    
    insights = f"""
    ### 📈 Performance Analysis
    
    - **EOQ Efficiency**: The calculated EOQ of {optimizer.eoq:,} units represents 
      {(optimizer.eoq/optimizer.annual_demand)*100:.1f}% of annual demand
    - **Service Level**: {"✅ Excellent - No stockouts" if stockout_months == 0 else f"⚠️ {stockout_months} stockout(s) occurred"}
    - **Inventory Turnover**: {optimizer.annual_demand/avg_inventory:.1f} times per year
    - **Demand Variability**: CV = {demand_cv:.2%}
    
    ### 💡 Recommendations
    
    1. **Current Performance**: The reorder point at {optimizer.reorder_percentage*100}% of EOQ 
       ({int(optimizer.reorder_point)} units) is {"adequate" if stockout_months == 0 else "insufficient"}
       
    2. **Safety Stock**: Consider adding {int(1.65 * df['Demand'].std()):.0f} units 
       of safety stock for 95% service level
       
    3. **Cost Optimization**: Average inventory of {avg_inventory:,.0f} units results in 
       ${avg_inventory * optimizer.holding_cost:,.2f} annual holding cost
    """
    
    return insights

# Main Streamlit App
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📦 Inventory Optimization Dashboard")
        st.markdown("*AI-Driven Retail Inventory Management using EOQ Model*")
    
    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        # Input parameters
        annual_demand = st.number_input(
            "📊 Annual Demand (units)",
            min_value=1000,
            max_value=1000000,
            value=12000,
            step=1000,
            help="Total expected demand for the year"
        )
        
        ordering_cost = st.number_input(
            "💰 Ordering Cost ($/order)",
            min_value=1,
            max_value=1000,
            value=50,
            step=5,
            help="Fixed cost per order placement"
        )
        
        holding_cost = st.number_input(
            "📦 Holding Cost ($/unit/year)",
            min_value=0.1,
            max_value=100.0,
            value=2.0,
            step=0.1,
            help="Cost to hold one unit for one year"
        )
        
        st.markdown("---")
        st.subheader("🎯 Simulation Settings")
        
        reorder_percentage = st.slider(
            "Reorder Point (% of EOQ)",
            min_value=10,
            max_value=50,
            value=25,
            step=5,
            help="Reorder when inventory drops below this percentage"
        ) / 100
        
        demand_variation = st.slider(
            "Demand Variation (%)",
            min_value=0,
            max_value=30,
            value=10,
            step=5,
            help="Monthly demand variation percentage"
        ) / 100
        
        simulation_months = st.selectbox(
            "Simulation Period",
            options=[6, 12, 24],
            index=1,
            help="Number of months to simulate"
        )
        
        st.markdown("---")
        
        # Run simulation button
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            st.session_state.simulation_run = True
    
    # Main content area
    if st.session_state.simulation_run:
        # Initialize optimizer
        optimizer = InventoryOptimizer(
            annual_demand=annual_demand,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            reorder_percentage=reorder_percentage
        )
        
        # Run simulation
        with st.spinner("Running inventory simulation..."):
            df, total_orders = optimizer.simulate_inventory(
                months=simulation_months,
                demand_variation=demand_variation
            )
        
        # Display KPIs
        st.markdown("### 📊 Key Performance Indicators")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric(
                label="Economic Order Quantity",
                value=f"{optimizer.eoq:,} units",
                delta="Optimal"
            )
        
        with kpi_col2:
            st.metric(
                label="Orders per Year",
                value=f"{optimizer.num_orders:.1f}",
                delta=f"{total_orders} actual"
            )
        
        with kpi_col3:
            st.metric(
                label="Total Annual Cost",
                value=f"${optimizer.total_annual_cost:,.2f}",
                delta=f"-{((1-(optimizer.total_annual_cost/(annual_demand*holding_cost)))*100):.1f}%"
            )
        
        with kpi_col4:
            st.metric(
                label="Avg Inventory Level",
                value=f"{df['Ending_Inventory'].mean():,.0f} units",
                delta=f"{(df['Ending_Inventory'].mean()/optimizer.eoq)*100:.0f}% of EOQ"
            )
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualization", "📊 Data Table", "💡 Insights", "📥 Export"])
        
        with tab1:
            # Main inventory plot
            fig_inventory = create_inventory_plot(df, optimizer.eoq, optimizer.reorder_point)
            st.plotly_chart(fig_inventory, use_container_width=True)
            
            # Cost breakdown
            col1, col2 = st.columns([2, 1])
            with col2:
                fig_cost = create_cost_breakdown_chart(optimizer, total_orders)
                st.plotly_chart(fig_cost, use_container_width=True)
            
            with col1:
                # Demand pattern visualization
                fig_demand = px.bar(
                    df,
                    x='Month',
                    y='Demand',
                    title='Monthly Demand Pattern',
                    color='Demand',
                    color_continuous_scale='Blues',
                    height=300
                )
                fig_demand.update_layout(showlegend=False)
                st.plotly_chart(fig_demand, use_container_width=True)
        
        with tab2:
            # Display simulation data
            st.markdown("### 📋 Monthly Simulation Results")
            
            # Format the dataframe for display
            display_df = df.copy()
            display_df['Order_Status'] = display_df.apply(
                lambda row: '✅ Ordered' if row['Order_Placed'] else '-', axis=1
            )
            display_df['Stock_Status'] = display_df.apply(
                lambda row: '⚠️ Stockout' if row['Stockout'] else '✅ In Stock', axis=1
            )
            
            # Select columns to display
            display_columns = ['Month', 'Starting_Inventory', 'Demand', 'Order_Status', 
                             'Order_Quantity', 'Ending_Inventory', 'Stock_Status']
            
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                height=400
            )
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Total Orders Placed**: {total_orders}")
            with col2:
                st.info(f"**Stockout Occurrences**: {len(df[df['Stockout']])}")
            with col3:
                st.info(f"**Avg Monthly Demand**: {df['Demand'].mean():,.0f} units")
        
        with tab3:
            # Display insights
            insights = generate_insights(optimizer, df, total_orders)
            st.markdown(insights)
            
            # Advanced analytics
            with st.expander("🔬 Advanced Analytics"):
                st.markdown("""
                ### 🎲 Monte Carlo Simulation Suggestion
                
                To better understand inventory risk, consider running a Monte Carlo simulation with:
                - 1000 iterations
                - Demand distribution: Normal(μ={:.0f}, σ={:.0f})
                - This would provide confidence intervals for stockout probability
                
                ### 📈 Demand Forecasting Integration
                
                Implement time-series forecasting using:
                - **ARIMA** for trend analysis
                - **Prophet** for seasonality detection
                - **Machine Learning** for pattern recognition
                """.format(df['Demand'].mean(), df['Demand'].std()))
        
        with tab4:
            # Export options
            st.markdown("### 📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Simulation Data (CSV)",
                    data=csv,
                    file_name=f"inventory_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Generate report summary
                report = f"""
                INVENTORY OPTIMIZATION REPORT
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                
                PARAMETERS:
                - Annual Demand: {annual_demand:,} units
                - Ordering Cost: ${ordering_cost}
                - Holding Cost: ${holding_cost}/unit/year
                - EOQ: {optimizer.eoq:,} units
                
                RESULTS:
                - Total Orders: {total_orders}
                - Total Annual Cost: ${optimizer.total_annual_cost:,.2f}
                - Average Inventory: {df['Ending_Inventory'].mean():,.0f} units
                - Stockouts: {len(df[df['Stockout']])}
                """
                
                st.download_button(
                    label="📑 Download Report (TXT)",
                    data=report,
                    file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to the Inventory Optimization Dashboard! 👋</h2>
            <p style='font-size: 18px; color: gray;'>
                Configure your parameters in the sidebar and click 
                <strong>'Run Optimization'</strong> to begin.
            </p>
            <br>
            <h3>🎯 What this app does:</h3>
            <ul style='text-align: left; display: inline-block;'>
                <li>Calculates optimal order quantity using EOQ model</li>
                <li>Simulates inventory levels over time</li>
                <li>Visualizes depletion and replenishment patterns</li>
                <li>Provides cost analysis and recommendations</li>
                <li>Exports results for further analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample visualization
        with st.expander("📊 See Sample Visualization"):
            # Create sample data
            sample_months = list(range(1, 13))
            sample_inventory = [1000, 850, 700, 550, 1400, 1250, 1100, 950, 800, 1650, 1500, 1350]
            
            fig_sample = go.Figure()
            fig_sample.add_trace(go.Scatter(
                x=sample_months,
                y=sample_inventory,
                mode='lines+markers',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8),
                name='Sample Inventory'
            ))
            
            fig_sample.update_layout(
                title="Sample Inventory Pattern",
                xaxis_title="Month",
                yaxis_title="Inventory Level",
                height=400
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)

if __name__ == "__main__":
    main()
