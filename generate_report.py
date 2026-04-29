import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# CONFIGURACIÓN 
# Si usaste .parquet en el paso 1, déjalo así. Si cambiaste a .csv, cambia el nombre.
INPUT_FILE = 'm5_analysis.parquet' 
METRICS_FILE = 'memory_metrics.csv'
OUTPUT_HTML = 'M5_Final_Story_Presentation.html'

def generate_report():
    print("Loading data... (This might take a minute)")
    try:
        # Si usas CSV, descomenta la línea de abajo y comenta la de read_parquet
        #df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
        
        # Si usas Parquet (recomendado):
        df = pd.read_parquet(INPUT_FILE)
             
        mem_metrics = pd.read_csv(METRICS_FILE)
    except FileNotFoundError:
        print("Please run step1_etl_processor.py first!")
        return

    charts = []
    
    # --- SECTION 1: THE TECHNICAL CHALLENGE ---
    print("Generating: Memory Optimization...")
    fig0 = px.bar(mem_metrics, x='Metric', y='Size_MB', 
                  title='1. Technical Challenge: Memory Optimization',
                  text='Size_MB', color='Metric', color_discrete_sequence=['#e74c3c', '#2ecc71'])
    fig0.update_traces(texttemplate='%{text:.0f} MB', textposition='outside')
    charts.append((fig0, "Start with the technical hurdle: Drastic reduction in memory usage enabled this analysis."))

    # --- SECTION 2: MACRO OVERVIEW ---
    print("Generating: Macro View...")
    
    # State
    state_rev = df.groupby('state_id')['revenue'].sum().reset_index()
    fig1 = px.bar(state_rev, x='state_id', y='revenue', title='2. Revenue by State', color_discrete_sequence=['#3498db'])
    charts.append((fig1, "California is the leading market."))

    # Stores
    store_rev = df.groupby('store_id')['revenue'].sum().sort_values(ascending=False).reset_index()
    fig2 = px.bar(store_rev, x='store_id', y='revenue', title='3. Revenue by Store', color_discrete_sequence=['#e67e22'])
    charts.append((fig2, "Store performance hierarchy."))

    # Category
    cat_rev = df.groupby('cat_id')['revenue'].sum().reset_index()
    fig3 = px.bar(cat_rev, x='cat_id', y='revenue', title='4. Revenue by Category', color_discrete_sequence=['teal'])
    charts.append((fig3, "Foods is the dominant category."))
    
    # Department
    dept_rev = df.groupby('dept_id')['revenue'].sum().sort_values(ascending=True).reset_index()
    fig4 = px.bar(dept_rev, x='revenue', y='dept_id', orientation='h', title='5. Revenue by Department', color_discrete_sequence=['purple'])
    charts.append((fig4, "Detailed department breakdown."))

    # --- SECTION 3: COMPOSITION (NORMALIZED 100%) ---
    print("Generating: Mix Analysis (Normalized to 100%)...")
    
    state_cat_mix = df.groupby(['state_id', 'cat_id'])['revenue'].sum().reset_index()
    fig5 = px.bar(state_cat_mix, x='state_id', y='revenue', color='cat_id', 
                  title='6. Category Mix by State (Normalized %)')
    fig5.update_layout(barnorm='percent') 
    fig5.update_yaxes(title='Percentage (%)') 
    charts.append((fig5, "CA and TX have a similar mix, but WI has 4% more sales in Foods."))
    
    state_dept_mix = df.groupby(['state_id', 'dept_id'])['revenue'].sum().reset_index()
    fig6 = px.bar(state_dept_mix, x='state_id', y='revenue', color='dept_id', 
                  title='7. Department Mix by State (Normalized %)')
    fig6.update_layout(barnorm='percent')
    fig6.update_yaxes(title='Percentage (%)')
    charts.append((fig6, "Detailed department mix comparison."))

    store_cat_mix = df.groupby(['store_id', 'cat_id'])['revenue'].sum().reset_index()
    fig7 = px.bar(store_cat_mix, x='store_id', y='revenue', color='cat_id', 
                  title='8. Category Mix by Store (Normalized %)')
    fig7.update_layout(barnorm='percent')
    fig7.update_yaxes(title='Percentage (%)')
    charts.append((fig7, "Detailed category mix comparison."))

    # --- SECTION 4: THE TRAP VS TRUTH (SEASONALITY) ---
    print("Generating: Seasonality...")
    
    df_trends = df[df['year'].isin([2012, 2013, 2014, 2015])].copy()
    seasonality = df_trends.groupby(['year', 'month'])['revenue'].sum().reset_index()
    
    # Trap
    fig8 = px.line(seasonality, x='month', y='revenue', color='year', markers=True,
                   title='9. Monthly Seasonality (Total Revenue) - The "Trap"')
    charts.append((fig8, "Notice the sharp drop in February (Short Month Effect)."))

    # Truth
    days_per_month = df_trends.groupby(['year', 'month'])['date'].nunique().reset_index(name='days')
    seasonality = pd.merge(seasonality, days_per_month, on=['year', 'month'])
    seasonality['daily_avg'] = seasonality['revenue'] / seasonality['days']
    
    fig9 = px.line(seasonality, x='month', y='daily_avg', color='year', markers=True,
                   title='10. Daily Average Revenue - The "Truth"')
    charts.append((fig9, "Normalizing by days reveals stable demand in Q1."))

    cat_season = df_trends.groupby(['year', 'month', 'cat_id'])['revenue'].sum().reset_index()
    fig10 = px.line(cat_season, x='month', y='revenue', color='year', facet_col='cat_id', 
                    title='11. Seasonality by Category', markers=True)
    charts.append((fig10, "Foods drives the Q4 volatility."))

    # --- SECTION 5: GRANULAR DRIVERS ---
    print("Generating: Granular Drivers...")
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    wday_rev = df.groupby('weekday')['revenue'].mean().reindex(days_order).reset_index()
    fig11 = px.bar(wday_rev, x='weekday', y='revenue', title='12. Average Revenue by Weekday', color_discrete_sequence=['#34495e'])
    charts.append((fig11, "The Weekend Effect."))

    df['is_event'] = df['event_name_1'].notnull()
    event_impact = df.groupby('is_event')['revenue'].mean().reset_index()
    event_impact['Label'] = event_impact['is_event'].map({True: 'Event Day', False: 'Normal Day'})
    fig12 = px.bar(event_impact, x='Label', y='revenue', title='13. Event Days vs Normal Days', color='Label')
    charts.append((fig12, "Trap: Normal days seem to drive higher average spend."))

    events = df[df['event_name_1'].notnull()]
    top_events = events.groupby('event_name_1')['revenue'].mean().sort_values(ascending=False).head(5).reset_index()
    fig13 = px.bar(top_events, x='event_name_1', y='revenue', title='14. Top 5 High-Impact Events')
    charts.append((fig13, "Truth: Super Bowl and Labor Day lead the pack."))

    events = df[df['event_name_1'].notnull()]
    bottom_events = events.groupby('event_name_1')['revenue'].mean().sort_values(ascending=True).head(5).reset_index()
    fig14 = px.bar(bottom_events, x='event_name_1', y='revenue', title='14. Bottom 5 High-Impact Events')
    charts.append((fig14, "Truth: Christmas and Thanksgiving are not labor days."))

    snap_impact = df.groupby(['cat_id', 'snap_active'])['revenue'].mean().reset_index()
    fig15 = px.bar(snap_impact, x='cat_id', y='revenue', color='snap_active', barmode='group',
                   title='15. SNAP Impact by Category')
    new_names = {True: 'SNAP Active', False: 'No SNAP'}
    
    # Safe rename for legend
    fig15.for_each_trace(lambda t: t.update(name = new_names.get(t.name == 'True', t.name))) 
    
    charts.append((fig15, "SNAP specifically boosts Food categories."))

    snap_cycle = df[df['cat_id']=='FOODS'].groupby(['state_id', 'day'])['revenue'].mean().reset_index()
    fig16 = px.line(snap_cycle, x='day', y='revenue', color='state_id', title='16. The Liquidity Cycle (Foods Only)')
    charts.append((fig16, "Sales peak during benefit issuance (Days 1-15)."))

    top_item = df.groupby('item_id')['revenue'].sum().idxmax()
    df_item = df[df['item_id'] == top_item]
    fig17 = px.scatter(df_item, x='sell_price', y='sales', opacity=0.5, 
                       title=f'17. Price Elasticity: Top Item ({top_item})')
    charts.append((fig17, "Analyzing price sensitivity for the top SKU."))

    # --- COMPILE HTML ---
    print(f"Compiling HTML Report to {OUTPUT_HTML}...")
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write("""<html><head><title>M5 Analysis Story</title>
        <style>body{font-family:sans-serif; margin:40px; background:#f4f4f9;} 
        .card{background:white; padding:20px; margin-bottom:40px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.1);}
        h1{color:#2c3e50; text-align:center;} .desc{color:#7f8c8d; font-style:italic; margin-bottom:15px;}</style>
        </head><body><h1>M5 Forecasting: From Data to Strategy</h1>""")
        
        for fig, desc in charts:
            f.write(f'<div class="card"><p class="desc">{desc}</p>')
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('</div>')
        
        f.write("</body></html>")
    
    print("DONE! Open the HTML file.")

if __name__ == "__main__":
    generate_report()