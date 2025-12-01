"""
Custom Python Visuals for Power BI - Contract Analytics
=======================================================
Generates matplotlib/plotly visualizations for embedding in Power BI.
Creates network graphs, timelines, heatmaps, and advanced charts.

Usage in Power BI:
    1. Add Python Visual to report
    2. Drag fields to Values
    3. Paste the relevant function from this file into Power BI's Python script editor
    4. Dataset will be available as 'dataset' pandas DataFrame
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, MonthLocator
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Power BI Dark Theme Colors
DARK_THEME = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'primary': '#00BCF2',      # Azure Blue
    'secondary': '#FFC107',    # Amber
    'success': '#4CAF50',      # Green
    'danger': '#F44336',       # Red
    'warning': '#FF9800',      # Orange
    'info': '#2196F3',         # Blue
    'grid': '#333333',
    'accent1': '#9C27B0',      # Purple
    'accent2': '#00BCD4',      # Cyan
}

def setup_dark_theme():
    """Configure matplotlib for dark theme"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': DARK_THEME['background'],
        'axes.facecolor': DARK_THEME['background'],
        'axes.edgecolor': DARK_THEME['grid'],
        'axes.labelcolor': DARK_THEME['text'],
        'text.color': DARK_THEME['text'],
        'xtick.color': DARK_THEME['text'],
        'ytick.color': DARK_THEME['text'],
        'grid.color': DARK_THEME['grid'],
        'figure.edgecolor': DARK_THEME['background'],
        'savefig.facecolor': DARK_THEME['background'],
        'savefig.edgecolor': DARK_THEME['background'],
    })


# ============================================================================
# VISUAL 1: Contract Expiration Timeline (Gantt-style)
# ============================================================================
def powerbi_contract_timeline():
    """
    Power BI Python Visual: Contract Expiration Timeline
    
    Required fields in Power BI:
    - document_name (Axis)
    - effective_date (Values)
    - expiration_date (Values)
    - status (Legend)
    - risk_score (Size - optional)
    """
    setup_dark_theme()
    
    # Power BI provides data as 'dataset'
    df = dataset.copy()
    
    # Convert dates
    df['effective_date'] = pd.to_datetime(df['effective_date'], errors='coerce')
    df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
    
    # Filter valid date ranges
    df = df.dropna(subset=['effective_date', 'expiration_date'])
    df['duration_days'] = (df['expiration_date'] - df['effective_date']).dt.days
    
    # Sort by expiration date
    df = df.sort_values('expiration_date').head(30)  # Top 30 for visibility
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color mapping for status
    status_colors = {
        'Active': DARK_THEME['success'],
        'Expired': DARK_THEME['danger'],
        'Terminated': DARK_THEME['warning'],
        'Draft': DARK_THEME['info'],
    }
    
    # Plot horizontal bars
    for idx, row in df.iterrows():
        start = row['effective_date']
        end = row['expiration_date']
        y_pos = idx
        
        color = status_colors.get(row.get('status', 'Active'), DARK_THEME['primary'])
        alpha = 0.9 if row.get('risk_score', 0) > 70 else 0.7
        
        ax.barh(y_pos, (end - start).days, left=start, height=0.8, 
                color=color, alpha=alpha, edgecolor='white', linewidth=0.5)
        
        # Add contract name
        ax.text(start, y_pos, f"  {row['document_name'][:40]}", 
                va='center', ha='left', fontsize=8, color=DARK_THEME['text'])
    
    # Add current date line
    today = pd.Timestamp.now()
    ax.axvline(today, color=DARK_THEME['secondary'], linestyle='--', linewidth=2, 
               label='Today', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Timeline', fontsize=12, fontweight='bold')
    ax.set_ylabel('Contracts', fontsize=12, fontweight='bold')
    ax.set_title('Contract Expiration Timeline\n(Ordered by Expiration Date)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    legend_elements = [mpatches.Patch(color=color, label=status, alpha=0.8) 
                       for status, color in status_colors.items()]
    legend_elements.append(plt.Line2D([0], [0], color=DARK_THEME['secondary'], 
                                     linewidth=2, linestyle='--', label='Today'))
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    plt.show()


# ============================================================================
# VISUAL 2: Party Network Graph (Relationship Map)
# ============================================================================
def powerbi_party_network():
    """
    Power BI Python Visual: Party Relationship Network
    
    Required fields in Power BI:
    - party_spectralink (Values)
    - party_counterparty (Values)
    - contract_type (Legend - optional)
    - company_folder (Tooltip - optional)
    """
    setup_dark_theme()
    
    df = dataset.copy()
    
    # Build network graph
    G = nx.Graph()
    
    # Add edges (relationships) between Spectralink and counterparties
    for idx, row in df.iterrows():
        spec = row.get('party_spectralink', 'Spectralink')
        counter = row.get('party_counterparty', 'Unknown')
        
        if counter != 'Unknown' and counter != '':
            G.add_edge(spec, counter, 
                      contract_type=row.get('contract_type', 'Unknown'),
                      weight=1)
    
    if len(G.nodes()) == 0:
        print("No relationships found in dataset")
        return
    
    # Calculate node sizes based on degree (number of connections)
    node_sizes = [G.degree(node) * 300 + 500 for node in G.nodes()]
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Identify Spectralink entities (central nodes)
    spectralink_nodes = [node for node in G.nodes() if 'spectralink' in node.lower()]
    other_nodes = [node for node in G.nodes() if 'spectralink' not in node.lower()]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color=DARK_THEME['grid'], 
                           width=1.5, ax=ax)
    
    # Draw Spectralink nodes (central)
    nx.draw_networkx_nodes(G, pos, nodelist=spectralink_nodes,
                           node_color=DARK_THEME['primary'], 
                           node_size=[G.degree(n) * 500 + 1000 for n in spectralink_nodes],
                           alpha=0.9, edgecolors='white', linewidths=2, ax=ax)
    
    # Draw counterparty nodes
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                           node_color=DARK_THEME['success'], 
                           node_size=[G.degree(n) * 300 + 500 for n in other_nodes],
                           alpha=0.7, edgecolors='white', linewidths=1, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_color=DARK_THEME['text'], 
                           font_weight='bold', ax=ax)
    
    ax.set_title('Contract Party Network\n(Node size = number of contracts)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# VISUAL 3: Risk Heatmap by Company & Contract Type
# ============================================================================
def powerbi_risk_heatmap():
    """
    Power BI Python Visual: Risk Score Heatmap
    
    Required fields in Power BI:
    - company_folder (Axis)
    - contract_type (Legend)
    - risk_score (Values)
    """
    setup_dark_theme()
    
    df = dataset.copy()
    
    # Create pivot table
    pivot = df.pivot_table(values='risk_score', 
                           index='company_folder', 
                           columns='contract_type', 
                           aggfunc='mean', 
                           fill_value=0)
    
    # Limit to top companies and types for readability
    pivot = pivot.head(20)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap with custom colormap
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                center=50, vmin=0, vmax=100,
                cbar_kws={'label': 'Risk Score'},
                linewidths=0.5, linecolor=DARK_THEME['grid'],
                ax=ax)
    
    ax.set_title('Contract Risk Scores by Company & Type\n(0=Low Risk, 100=High Risk)', 
                 fontsize=14, fontweight='bold', pad=20, color=DARK_THEME['text'])
    ax.set_xlabel('Contract Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Company', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ============================================================================
# VISUAL 4: GDPR Compliance Treemap
# ============================================================================
def powerbi_gdpr_treemap():
    """
    Power BI Python Visual: GDPR Compliance Distribution
    
    Required fields in Power BI:
    - region (Axis)
    - gdpr_applicable (Legend)
    - contract_id (Values - Count)
    """
    setup_dark_theme()
    
    df = dataset.copy()
    
    # Group by region and GDPR status
    summary = df.groupby(['region', 'gdpr_applicable']).size().reset_index(name='count')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prepare data for treemap
    regions = summary['region'].unique()
    colors = []
    sizes = []
    labels = []
    
    for region in regions:
        region_data = summary[summary['region'] == region]
        for idx, row in region_data.iterrows():
            sizes.append(row['count'])
            gdpr_status = 'GDPR' if row['gdpr_applicable'] else 'Non-GDPR'
            labels.append(f"{region}\n{gdpr_status}\n{row['count']} contracts")
            
            # Color based on GDPR status
            if row['gdpr_applicable']:
                colors.append(DARK_THEME['warning'])
            else:
                colors.append(DARK_THEME['success'])
    
    # Use squarify for treemap (if available, else fallback to pie)
    try:
        import squarify
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, 
                     text_kwargs={'fontsize':9, 'color':DARK_THEME['text'], 'weight':'bold'},
                     ax=ax)
        ax.axis('off')
    except ImportError:
        # Fallback to pie chart
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, textprops={'color':DARK_THEME['text']})
    
    ax.set_title('GDPR Compliance by Region\n(Contract Distribution)', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


# ============================================================================
# VISUAL 5: Contract Status Funnel
# ============================================================================
def powerbi_status_funnel():
    """
    Power BI Python Visual: Contract Lifecycle Funnel
    
    Required fields in Power BI:
    - status (Axis)
    - contract_id (Values - Count)
    """
    setup_dark_theme()
    
    df = dataset.copy()
    
    # Count by status
    status_counts = df['status'].value_counts()
    
    # Define funnel stages (ordered)
    funnel_order = ['Draft', 'Active', 'Expiring Soon', 'Expired', 'Terminated']
    funnel_data = []
    
    for stage in funnel_order:
        if stage in status_counts.index:
            funnel_data.append((stage, status_counts[stage]))
        elif stage == 'Expiring Soon' and 'is_expiring_soon' in df.columns:
            count = df['is_expiring_soon'].sum()
            if count > 0:
                funnel_data.append((stage, count))
    
    if not funnel_data:
        funnel_data = [(status, count) for status, count in status_counts.items()]
    
    stages, counts = zip(*funnel_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color scheme
    colors = [DARK_THEME['info'], DARK_THEME['success'], DARK_THEME['warning'], 
              DARK_THEME['danger'], DARK_THEME['accent1']][:len(stages)]
    
    # Create horizontal funnel
    y_pos = np.arange(len(stages))
    max_width = max(counts)
    
    for i, (stage, count) in enumerate(zip(stages, counts)):
        width = count / max_width
        ax.barh(i, width, height=0.7, color=colors[i], alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add labels
        ax.text(width/2, i, f"{stage}\n{count:,} contracts", 
                ha='center', va='center', fontsize=11, fontweight='bold', 
                color=DARK_THEME['text'])
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1.1)
    ax.set_title('Contract Lifecycle Funnel', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# VISUAL 6: Expiration Calendar Heatmap
# ============================================================================
def powerbi_expiration_calendar():
    """
    Power BI Python Visual: Expiration Date Calendar Heatmap
    
    Required fields in Power BI:
    - expiration_date (Values)
    - contract_id (Values - Count)
    """
    setup_dark_theme()
    
    df = dataset.copy()
    df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
    df = df.dropna(subset=['expiration_date'])
    
    # Group by month
    df['month'] = df['expiration_date'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    
    # Create date range for next 24 months
    today = pd.Timestamp.now()
    future_months = pd.period_range(start=today, periods=24, freq='M')
    
    # Build calendar grid (12 months x 2 years)
    calendar_data = np.zeros((2, 12))
    
    for period in future_months:
        if period in monthly_counts.index:
            year_offset = period.year - today.year
            month_idx = period.month - 1
            if 0 <= year_offset < 2:
                calendar_data[year_offset, month_idx] = monthly_counts[period]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(calendar_data, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                yticklabels=[f'{today.year}', f'{today.year+1}'],
                cbar_kws={'label': 'Contracts Expiring'},
                linewidths=1, linecolor=DARK_THEME['grid'],
                ax=ax)
    
    ax.set_title('Contract Expiration Calendar\n(Next 24 Months)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Year', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Example usage (for testing outside Power BI)
# ============================================================================
if __name__ == '__main__':
    # Create sample dataset for testing
    dataset = pd.DataFrame({
        'document_name': [f'Contract_{i}' for i in range(50)],
        'effective_date': pd.date_range(start='2023-01-01', periods=50, freq='W'),
        'expiration_date': pd.date_range(start='2025-01-01', periods=50, freq='W'),
        'status': np.random.choice(['Active', 'Expired', 'Terminated'], 50),
        'risk_score': np.random.randint(0, 100, 50),
        'party_spectralink': ['Spectralink Corp'] * 50,
        'party_counterparty': [f'Company_{i%10}' for i in range(50)],
        'contract_type': np.random.choice(['NDA', 'MSA', 'SOW', 'License'], 50),
        'company_folder': [f'Company_{i%20}' for i in range(50)],
        'region': np.random.choice(['AMER', 'EMEA', 'APAC'], 50),
        'gdpr_applicable': np.random.choice([True, False], 50),
        'is_expiring_soon': np.random.choice([True, False], 50),
    })
    
    print("Testing visual generators...")
    print("Run these functions individually in Power BI Python Visual")
