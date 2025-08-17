import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from datetime import time
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.io as pio
import plotly.express as px
import os
import plotly.graph_objects as go
import pytz
import nltk
nltk.download("vader_lexicon", quiet=True)




apps_df = pd.read_csv("Play Store Data.csv")
reviews_df = pd.read_csv("User Reviews.csv")

# preprocessing
apps_df = apps_df.dropna(subset=['Rating']).copy()
for column in apps_df.columns:
    apps_df[column] = apps_df[column].fillna(apps_df[column].mode()[0])
apps_df.drop_duplicates(inplace=True)
apps_df=apps_df[apps_df['Rating']<=5].copy()
reviews_df.dropna(subset=['Translated_Review'], inplace=True)
apps_df = apps_df[~apps_df['Installs'].isin(['Free', '0', '0+'])].copy()
apps_df['Installs'] = apps_df['Installs'].astype(str).str.replace(',', '').str.replace('+', '').astype(int)
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)
merged_df = pd.merge(apps_df, reviews_df, on ='App', how='inner')
def convert_size(size):
    if isinstance(size, str):
        size = size.strip().upper()
        if size.endswith('M'):
            try:
                return float(size[:-1])  # remove 'M' and convert to float
            except ValueError:
                return np.nan
        elif size.endswith('K'):
            try:
                return float(size[:-1]) / 1024  # convert kB to MB
            except ValueError:
                return np.nan
        elif 'VARIES' in size:
            return np.nan
    return np.nan  # For floats or NaNs
apps_df['Size'] = apps_df['Size'].apply(convert_size)
apps_df['Log_Installs'] = np.log(apps_df['Installs'])
apps_df['Reviews']=apps_df['Reviews'].astype(int)
apps_df['Log_Reviews']=np.log(apps_df['Reviews'])
def rating_group(Rating):
    if Rating >= 4:
        return 'Top Rated App'
    elif Rating >=3:
        return 'Above Average'
    elif Rating >=2:
        return 'Average'
    else:
        return 'Below Average'

apps_df['Rating_group'] = apps_df['Rating'].apply(rating_group)
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']

# sentiment analysis
sia = SentimentIntensityAnalyzer()
reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Year']=apps_df['Last Updated'].dt.year


# preparing directory for html files
# creating a path to store our plotly graphs
html_files_path = "./"
# making use of operating system's library
if not os.path.exists(html_files_path):
    # makedirs means to make a directory/folder
    os.makedirs(html_files_path)

# plot_containers is like a box where you'll be saving the plot
plot_containers=""

def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    
    plot_containers += f'''
        <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
            <div class="plot">{html_content}</div>
            <div class="insights">{insight}</div>
        </div>
    '''
    
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')

plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}

# 1.) To visualize the top 10 app categories in the Google Play Store
category_counts = apps_df['Category'].value_counts().nlargest(10)

# Generate a bar graph
fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
     
      
)

# Customize layout
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=12)   
    ),
    yaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=12)   
    ),

    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=18)
    ),
    legend_title=dict(
        text="Color",
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)


# fig2: type analysis plot
# analyzing distribution of free vs paid apps
# since we're analyzing a categorical distribution with less than 5 or 6 categories, we use a pie chart
type_counts = apps_df['Type'].value_counts()

# Generate a pie chart
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    color_discrete_sequence=px.colors.sequential.RdBu,
    # textinfo='percent+label',  # Show both percentage and label
    # textfont=dict(size=20)  
      
)
fig2.update_traces(
    textfont=dict(size=18),
    textinfo='percent+label'
)
# Customize layout
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=18)
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)


# building a histogram
# rating distribution 
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    # title="Rating Distribution",
    color_discrete_sequence=['#636EFA'],
     
      
)

# Customize layout
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    # title_font=dict(size=16),
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),

    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=18)
    ),
    legend_title=dict(
        text="Color",
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# 4.) Displaying sentiment analysis using bar chart 
sentiment_counts = reviews_df['Sentiment_Score'].value_counts()

# Generate a bar graph
fig4 = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
     
      
)

# Customize layout
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),

    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=18)
    ),
    legend_title=dict(
        text="Color",
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Generate a bar graph
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)

fig5 = px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',  # horizontal bar chart 
    labels={'x': 'Installs', 'y': 'Category'},
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
     
      
)

# Customize layout
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18),
        domain=[0,0.8]  
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),
    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=18),
        x=0.85,
        y=1
    ),
    legend_title=dict(
        text="Color",
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# figure6
# Generate a line graph

updates_per_year = apps_df.groupby('Category')['Installs'].sum().nlargest(10)


fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    color_discrete_sequence=['#AB63FA'],
     
      
)

# Customize layout
fig6.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),

    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=18)
    ),
    legend_title=dict(
        text="Color",
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Generate a bar graph
# comparing the revenue generated by the app category
revenue_by_category= apps_df.groupby('Category')['Revenue'].sum().nlargest(10)


fig7 = px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
     
      
)

# Customize layout
fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),

    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=18)
    ),
    legend_title=dict(
        text="Color",
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# figure 8 
# Generate a bar graph
# to count the genre
# to visualize the top10 most common genre of apps
genre_counts= apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)


fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
     
      
)

# Customize layout
fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),

    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=20)
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# figure9 -- scatter plot
# analyze relationship between the last update date and the app ratings

fig9 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    color_discrete_sequence=px.colors.qualitative.Vivid,
     
      
)

# Customize layout
fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=18)   
    ),

    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=18),
        title_font=dict(size=20)
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# figure 10
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',   # color assigned based on the type
    color_discrete_sequence=px.colors.qualitative.Pastel,
     
      
)

fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=20)   
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=20)   
    ),
    # Increase legend font size (color values on right)
    legend=dict(
        font=dict(size=20),
        title_font=dict(size=20)
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)



# figure11 -- scatter plot
# visualize the relationship between revenue and the number of installs for paid apps
paid_apps_trimmed = apps_df[
    (apps_df['Type'] == 'Paid') &  
    (apps_df['Installs'] > 0) &    
    (apps_df['Revenue'] > 0)       
].copy()


paid_apps_trimmed = paid_apps_trimmed[['App', 'Category', 'Installs', 'Revenue']]


paid_apps_trimmed = paid_apps_trimmed[
    (paid_apps_trimmed['Installs'] <= paid_apps_trimmed['Installs'].quantile(0.99)) &
    (paid_apps_trimmed['Revenue'] <= paid_apps_trimmed['Revenue'].quantile(0.99))
]

fig11 = px.scatter(
    paid_apps_trimmed,
    x='Installs',
    y='Revenue',
    color='Category',
    color_discrete_sequence=px.colors.qualitative.Dark2,
     
      
)


x = np.log10(paid_apps_trimmed['Installs'])
y = np.log10(paid_apps_trimmed['Revenue'])
coeffs = np.polyfit(x, y, 1)  # slope & intercept
poly_eqn = np.poly1d(coeffs)


x_line = np.linspace(x.min(), x.max(), 100)
y_line = poly_eqn(x_line)


x_line_orig = 10**x_line
y_line_orig = 10**y_line


fig11.add_trace(go.Scatter(
    x=x_line_orig,
    y=y_line_orig,
    mode='lines',
    name='Overall Trendline',
    line=dict(color='white', width=1)
))


fig11.update_layout(
    xaxis_type='log',
    yaxis_type='log',
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    # xaxis=dict(title_font=dict(size=12), domain=[0,0.82]),
    # yaxis=dict(title_font=dict(size=12)),   

    xaxis=dict(
        title_font=dict(size=25),
        tickfont=dict(size=20),
        domain=[0,0.8]  
    ),
    yaxis=dict(
        title_font=dict(size=25),
        tickfont=dict(size=20)   
    ),
    # Increase legend font size (color values on right)
    legend=dict(
        x=0.85,
        y=1,
        font=dict(size=12),
        title_font=dict(size=12)
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    legend_title_text='Category',
    # legend=dict(
    #     x=0.85,
    #     y=1,
    #     bgcolor="rgba(0,0,0,0)"
    # )
)

fig11.layout.xaxis.update(domain=[0, 0.85]) 


## figure 12

# Function to convert size to MB
def size_to_mb(size):
    if isinstance(size, str):
        size = size.strip().upper()
        if size.endswith('M'):
            return float(size.replace('M', ''))
        elif size.endswith('K'):
            return float(size.replace('K', '')) / 1024
        elif size.endswith('G'):
            return float(size.replace('G', '')) * 1024
    return float(size) if isinstance(size, (int, float)) else 0.0

# Convert sizes
apps_df['Size_MB'] = apps_df['Size'].apply(size_to_mb)

# Apply filters
apps2_df = apps_df[
    (apps_df['Installs'] >= 10000) &
    (apps_df['Revenue'] >= 10000) &
    (apps_df['Android Ver'].apply(
        lambda x: float(x.split()[0]) if isinstance(x, str) and x.split()[0].replace('.', '', 1).isdigit() else 0
    ) > 4.0) &
    (apps_df['Size_MB'] > 15.0) &
    (apps_df['Content Rating'] == 'Everyone') &
    (apps_df['App'].apply(lambda x: len(str(x)) <= 30))
]

# Top 3 categories by installs
top_categories = (
    apps2_df.groupby('Category')['Installs']
    .sum()
    .sort_values(ascending=False)
    .head(3)
    .index
    .tolist()
)

df_top3 = apps2_df[apps2_df['Category'].isin(top_categories)]

# Group data
grouped_data = df_top3.groupby(['Category', 'Type']).agg({
    'Installs': 'mean',
    'Revenue': 'mean'
}).reset_index()

free = grouped_data[grouped_data['Type'] == 'Free'].set_index('Category').reindex(top_categories).fillna(0)
paid = grouped_data[grouped_data['Type'] == 'Paid'].set_index('Category').reindex(top_categories).fillna(0)

# Check time (IST)
def is_time_in_ist_range_1pm_2pm():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist).time()
    return time(13, 0) <= now_ist < time(14, 0)

# Build fig12 only if in range
fig12 = None
if is_time_in_ist_range_1pm_2pm():
    fig12 = go.Figure()

    # Avg Installs
    fig12.add_trace(go.Bar(
        x=top_categories,
        y=free['Installs'],
        name='Free - Avg Installs',
        marker_color='royalblue',
        offsetgroup=0
    ))

    fig12.add_trace(go.Bar(
        x=top_categories,
        y=paid['Installs'],
        name='Paid - Avg Installs',
        marker_color='deepskyblue',
        offsetgroup=1
    ))

    # Avg Revenue (second axis)
    fig12.add_trace(go.Bar(
        x=top_categories,
        y=free['Revenue'],
        name='Free - Avg Revenue',
        marker_color='limegreen',
        yaxis='y2',
        offsetgroup=2
    ))

    fig12.add_trace(go.Bar(
        x=top_categories,
        y=paid['Revenue'],
        name='Paid - Avg Revenue',
        marker_color='darkgreen',
        yaxis='y2',
        offsetgroup=3
    ))

    fig12.update_layout(
        xaxis=dict(
            title='App Category',
            title_font=dict(size=25),
            tickfont=dict(size=20),
            domain=[0,0.7]
        ),
        yaxis=dict(
            title='Average Installs',
            title_font=dict(size=22),
            tickfont=dict(size=18)
        ),
        yaxis2=dict(
            title='Average Revenue',
            overlaying='y',
            side='right',
            title_font=dict(size=22),
            tickfont=dict(size=18)
        ),
        legend=dict(
            font=dict(size=20),           
            title_font=dict(size=22), 
            x=0.86,
            y=1,
            xanchor='left',
            yanchor='top',
        ),
        barmode='group',
        paper_bgcolor='black',
        font_color='white'
    )


## figure 14

# Ensure datetime type
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'])

# Filters
apps3_df = apps_df[
    (~apps_df['App'].str.lower().str.startswith(('x', 'y', 'z'))) &
    (apps_df['Category'].str.upper().str.startswith(('E', 'C', 'B'))) &
    (apps_df['Reviews'] > 500) &
    (~apps_df['App'].str.lower().str.contains('s'))
].copy()

# Translation map
translation_map = {
    'Beauty': 'सौंदर्य',       # Hindi
    'Business': 'வணிகம்',      # Tamil
    'Dating': 'Verabredung'    # German
}

def translate_category(cat):
    cat_lower = str(cat).lower()
    if cat_lower == 'beauty':
        return translation_map['Beauty']
    elif cat_lower == 'business':
        return translation_map['Business']
    elif cat_lower == 'dating':
        return translation_map['Dating']
    else:
        return cat

apps3_df['Category_Translated'] = apps3_df['Category'].apply(translate_category)

# Group by month
apps3_df['Month'] = apps3_df['Last Updated'].dt.to_period('M').dt.to_timestamp()
grouped = apps3_df.groupby(['Month', 'Category_Translated']).agg({'Installs': 'sum'}).reset_index()

# Previous month installs
grouped['Prev_Installs'] = grouped.groupby('Category_Translated')['Installs'].shift(1)
grouped['MoM_growth'] = (grouped['Installs'] - grouped['Prev_Installs']) / grouped['Prev_Installs']


def is_time_in_ist_range():
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = datetime.now(ist).time().replace(microsecond=0)  # strip microseconds
    return time(18, 0) <= ist_now <= time(21, 0)

fig14 = None

if is_time_in_ist_range():
    fig14=go.Figure()
    categories = grouped['Category_Translated'].unique()

    for cat in categories:
        cat_data = grouped[grouped['Category_Translated'] == cat]

        fig14.add_trace(go.Scatter(
            x=cat_data['Month'],
            y=cat_data['Installs'],
            mode='lines+markers',
            name=cat
        ))

        growth_highlights = cat_data[cat_data['MoM_growth'] > 0.20]
        for i in range(1, len(growth_highlights)):
            prev_row = growth_highlights.iloc[i - 1]
            row = growth_highlights.iloc[i]
            fig14.add_trace(go.Scatter(
                x=[prev_row['Month'], row['Month'], row['Month'], prev_row['Month']],
                y=[0, 0, row['Installs'], prev_row['Installs']],
                fill='toself',
                fillcolor='rgba(255, 200, 200, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))

    fig14.update_layout(
        xaxis=dict(
            title='Month',
            title_font=dict(size=25),
            tickfont=dict(size=20),
            # domain=[0,0.7]
        ),
        yaxis=dict(
            title='Total Installs',
            title_font=dict(size=22),
            tickfont=dict(size=18)
        ),
        legend=dict(
            title='App Category',
            font=dict(size=20),           
            title_font=dict(size=22)
        ),
        hovermode='x unified',
        paper_bgcolor='black',
        font_color='white'
    )

else:
    print("Graph visible only between 6 PM and 9 PM IST.")


st.set_page_config(page_title="Google Play Store Analytics", layout="wide")

# inject css
st.markdown(
    """
    <style>
    /* Background color for entire page */
    body {
        font-family: Arial, sans-serif;
        background-color: green;
        color: #fff;
        margin: 0;
        padding: 0;
    }

    .stApp {
        background-color: onyx;
        color: white;
    }

    .header{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background-color: #444;
        margin-top: 0px;
    }

    .header img{
        margin: 0 10px;
        height: 50px;
    }

    
    </style>
    """,
    unsafe_allow_html=True
)



# html layout
st.markdown(
    """
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google" width="180">
        <h1 style="color: white; margin: 0;">Google Play Store Reviews Analytics</h1>
        <img src="https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png" alt="Google Play" width="200" height="300">
    </div>


    """,
    unsafe_allow_html=True
)

# Inject responsive CSS
st.markdown(
    """
    <style>
    /* Make images and headings responsive */
    .header img {
    max-width: 100%;
    height: auto;
    }

    /* Adjust heading font sizes for smaller screens */
    @media (max-width: 768px) {
    h1 {
        font-size: 22px !important;
    }
    h2, h3 {
        font-size: 18px !important;
    }
    p {
        font-size: 16px !important;
    }
    }

    /* Improve plot container responsiveness */
    .plot-container {
    width: 100% !important;
    overflow-x: auto;
    }

    /* Stack columns vertically on mobile */
    @media (max-width: 768px) {
    .stColumns {
        display: block !important;
    }
    .stColumn {
        width: 100% !important;
        margin-bottom: 20px;
    }
    }
    </style>
    """, 
    unsafe_allow_html=True
)

def styled_heading(text):
    st.markdown(
        f"""
        <h1 style="font-size:20px; color:white; text-align:center;">
            {text}
        </h1>
        """,
        unsafe_allow_html=True
    )

def styled_insight(text):
    st.markdown(
        f"""
        <p style="font-size:18px; color:#d3d3d3; line-height:1.5; text-align:center;">
            💡 <b>Insight:</b> {text}
        </p>
        """,
        unsafe_allow_html=True
    )




col1, col2, col3 = st.columns(3)


with col1:
    styled_heading("Top Categories on Play Store")
    st.plotly_chart(fig1, use_container_width=True)
    styled_insight("The Top Categories on Play Store are dominated by tools, entertainment, and productivity apps.")

with col2:
    styled_heading("App Type Distribution")
    st.plotly_chart(fig2, use_container_width=True)
    styled_insight("Most Apps on PlayStore are free, indicating a strategy to attract users first and monetize through Ads or in App Purchases.")

with col3:
    styled_heading("Rating Distribution")
    st.plotly_chart(fig3, use_container_width=True)
    styled_insight("Ratings are skewed towards higher value, suggesting that most apps are rated favorably by users.")

st.write("---")
col4, col5, col6 = st.columns(3)

with col4:
    styled_heading("Sentiment Distribution")
    st.plotly_chart(fig4, use_container_width=True)
    styled_insight("Sentiment in Reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments.")

with col5:
    styled_heading("Installs By Category")
    st.plotly_chart(fig5, use_container_width=True)
    styled_insight("The Categories with the most installs are social and communication apps, reflecting their broad appeal and daily usage.")

with col6:
    styled_heading("Number of Updates over the Years")
    st.plotly_chart(fig6, use_container_width=True)
    styled_insight("Updates have been increasing over the years, showing that developers have been actively maintaining and improving their apps.")

st.write("---")
col7, col8, col9 = st.columns(3)

with col7:
    styled_heading("Revenue by Category")
    st.plotly_chart(fig7, use_container_width=True)
    styled_insight("Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential.")

with col8:
    styled_heading("Revenue by Category")
    st.plotly_chart(fig8, use_container_width=True)
    styled_insight("Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games.")

with col9:
    styled_heading("Impact of Last Update on Rating")
    st.plotly_chart(fig9, use_container_width=True)
    styled_insight("The Scatter Plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates don't always result in better Ratings.")


st.write("---")
col10, col11, col12 = st.columns(3)

with col10:
    styled_heading("Rating for Paid vs Free Apps")
    st.plotly_chart(fig10, use_container_width=True)
    styled_insight("Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from the apps they pay for.")

with col11:
    styled_heading("Revenue Trends by Paid App Installs")
    st.plotly_chart(fig11, use_container_width=True)
    styled_insight("User engagement peaks during the week, drops on weekends, and is highest in the morning. Thursdays and Fridays see the most activity, with a midday activity dip.")

with col12:
    styled_heading("Avg Installs & Revenue for Top 3 Categories")
    if fig12 is not None:
        st.plotly_chart(fig12, use_container_width=True)
        styled_insight("Free apps have significantly higher average installs across all categories, while paid apps generate more revenue per app, especially in the Photography category.")
    else:
        st.warning("Graph visible only between 1 PM and 2 PM IST.")


st.write("---")
col13, col14, col15 = st.columns(3)

with col14:
    styled_heading("Trend of Total Installs Over Time by App Category")
    if fig14 is not None:
        st.plotly_chart(fig14, use_container_width=True, key="fig14_chart")
        styled_insight("Communication apps saw a massive spike in installs around 2018, far surpassing other categories.")
    else:
        st.warning("Graph visible only between 6 PM and 9 PM IST.")
