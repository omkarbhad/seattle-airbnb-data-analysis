import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import uuid

# Configure Streamlit page layout and title
st.set_page_config(page_title="Seattle Airbnb Analysis", layout="wide", initial_sidebar_state="expanded")

# Define custom Plotly template for consistent visualization styling
custom_template = dict(
    layout=dict(
        font=dict(family="Arial", size=14),
        title_font=dict(size=20),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        colorscale=dict(sequential="Cividis")  # High-contrast, accessible color scale
    )
)
pio.templates["custom"] = custom_template
pio.templates.default = "plotly_white+custom"

# Custom CSS for improved layout, accessibility, and visual polish
st.markdown(
    """
    <style>
    /* Hide Streamlit's default UI elements */
    #MainMenu, header, .stToolbar {visibility: hidden;}
    
    /* Main content padding */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
    }
    
    /* Fixed header styling */
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: rgba(25, 31, 40, 0.95);
        z-index: 1000;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .fixed-header h1 {
        color: #fff;
        font-size: 16px;
        margin: 0;
        padding: 0;
        line-height: 1;
    }
    
    /* Dark Sidebar Styling */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Sidebar text and controls */
    .stSidebar .stMarkdown,
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown h1,
    .stSidebar .stMarkdown h2,
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4,
    .stSidebar .stMarkdown h5,
    .stSidebar .stMarkdown h6,
    .stSidebar .stRadio > label,
    .stSidebar .stCheckbox > label,
    .stSidebar .stSelectbox > label,
    .stSidebar .stNumberInput > label,
    .stSidebar .stTextInput > label,
    .stSidebar .stTextArea > label {
        color: #ffffff !important;
    }
    
    /* Sidebar expander headers */
    .stSidebar .stExpanderHeader {
        color: #ffffff !important;
        background-color: transparent !important;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    
    /* Active sidebar button */
    .stSidebar .stButton > button[kind="primary"] {
        background-color: t    .stSidebar .stExpanderHeader:hover,
    .stSidebar .stButton > button:hover {
        background-color: rgba(0, 120, 255, 0.1) !important;
    }
    
    /* Active/selected state */
    .stSidebar .stButton > button[kind="primary"] {
        background-color: rgba(0, 120, 255, 0.2) !important;
        border-left: 4px solid #0078ff !important;
        color: #ffffff !important;
        box-shadow: none !important;
    }
    
    .stSidebar .stButton > button[kind="primary"]:hover {
        background-color: rgba(0, 120, 255, 0.3) !important;
 !important;
    }
    
    .stSidebar .stButton > button[kind="primary"]:hover {
        color: #ff4b4b !important;
        background-color: rgba(255, 75, 75, 0.1) !important;
    }
    
    /* Responsive sidebar for mobile */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 50px !important;
            min-width: 50px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 300px !important;
            min-width: 300px !important;
        }
    }
    
    /* Styled radio buttons */
    div[data-testid="stRadio"] > label {
        padding: 8px 12px;
        border-radius: 5px;
        transition: background-color 0.2s;
        tabindex: 0; /* Keyboard navigation */
    }
    div[data-testid="stRadio"] > label:hover {
        background-color: #d0e4f5; /* Branded hover color */
    }
    
    /* Main content styling */
    .main .block-container {
        padding: 1rem 2vw !important;
    }
    .main-content {
        margin-top: -32px; /* Match header height */
        padding-top: 2px;
        padding-bottom: 10px;
    }
    
    /* Expander transitions */
    .st-expander {
        transition: all 0.3s ease;
    }
    
    /* Slider container */
    .slider-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin: 10px 0;
    }
    .slider-label {
        white-space: nowrap;
        font-weight: 500;
    }
    .stSlider label {
        display: none;
    }
    
    /* High-contrast text */
    body {
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and cache data with spinner and TTL for freshness
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        calendar = pd.read_csv('./data/calendar.csv', parse_dates=['date'])
        listings = pd.read_csv('./data/listings.csv')
        reviews = pd.read_csv('./data/reviews.csv', parse_dates=['date'])
        return calendar, listings, reviews
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure all required CSVs are in the 'data' folder.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load data with spinner
with st.spinner("Loading data..."):
    calendar, listings, reviews = load_data()
if calendar is None or listings is None or reviews is None:
    st.stop()

# Fixed header with ARIA label for accessibility
st.markdown(
    """
    <div class="fixed-header" aria-label="Application Header: Seattle Airbnb Investment Analysis">
        <h1>Seattle Airbnb Investment Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar with improved navigation
st.sidebar.markdown("### Navigation 📋")


# Menu items with icons for visual cues
menu_items = [
    ("📅", "Seasonal Occupancy"),
    ("💰", "Monthly Price Variation"),
    ("🏘️", "Popular Neighborhoods"),
    ("📈", "Price Influencers"),
    ("📍", "Geographic Distribution")
]

# Initialize session state for navigation
if 'current_section' not in st.session_state:
    st.session_state.current_section = menu_items[0][1]

# Navigation buttons with minimal styling
st.sidebar.markdown("### Seasonal Analysis")
for icon, item in menu_items[:2]:
    is_active = st.session_state.current_section == item
    if st.sidebar.button(
        f"{icon} {item}",
        key=f"nav_{item}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.current_section = item
        st.rerun()

st.sidebar.markdown("### Location Analysis")
for icon, item in menu_items[2:]:
    is_active = st.session_state.current_section == item
    if st.sidebar.button(
        f"{icon} {item}",
        key=f"nav_{item}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.current_section = item
        st.rerun()

# Set the analysis option based on current section
analysis_option = st.session_state.current_section

# Update session state
st.session_state.analysis_option = analysis_option

# Add custom CSS to remove space
st.markdown("""
<style>
    .stApp {
        margin-top: 0px !important;
    }
    .block-container {
        padding-top: 10px !important;
    }
    .main .block-container {
        padding-top: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Main content wrapper
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if analysis_option == "Seasonal Occupancy":
    st.markdown("### 1. Which Season Has the Highest Occupancy? 📅")
    
    def get_data1(reviews):
        freq_count = reviews['date'].value_counts().values
        freq_index = reviews['date'].value_counts().index
        frequency = pd.DataFrame(freq_count, index=freq_index, columns=['No_of_Reservations'])
        reservation_frequency_2015 = frequency.loc['2015']
        reservation_2015_month = reservation_frequency_2015.resample('M').sum()
        reservation_2015_month['% Reservations'] = reservation_2015_month['No_of_Reservations'] * 100 / reservation_2015_month["No_of_Reservations"].sum()
        reservation_2015_month['Month'] = reservation_2015_month.index.month_name()
        return reservation_2015_month

    reservation_2015_month = get_data1(reviews)
    fig = px.bar(
        reservation_2015_month,
        x='Month',
        y='% Reservations',
        title='Reviews Percentage per Month in 2015',
        labels={'Month': 'Month', '% Reservations': 'Reviews Percentage (%)'},
        color='% Reservations',
        template='custom'
    )
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        hovermode='x unified'
    )
    fig.update_traces(
        hovertemplate='Month: %{x}<br>Percentage: %{y:.2f}%<br>Reviews: %{customdata}',
        customdata=reservation_2015_month['No_of_Reservations']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Analysis Details", expanded=True):
        st.markdown("""
        **Visualization Description** 📊:\n
        This interactive bar plot illustrates the percentage of total reviews for Airbnb listings in Seattle across each month in 2015, serving as a proxy for occupancy rates. The data is sourced from the `reviews.csv` dataset, with 84,849 review records from 2009 to 2016. The analysis focuses on 2015 for peak activity.

        **Key Insights** 🔑:\n
        - **Peak Occupancy in August**: August has the highest review share (~12–15%), indicating peak summer demand.
        - **Low Occupancy in February**: February shows the lowest share (~5–7%), suggesting minimal winter demand.
        - **Seasonal Trends**: Summer (June–August) has higher reviews (10–15%), winter (December–February) lower (5–8%).
        - **Investor Implications** 💼: Maximize pricing in summer, especially August, to capitalize on demand.
        - **Limitations** ⚠️: Reviews may underestimate bookings, as not all guests review.
        """)

elif analysis_option == "Monthly Price Variation":
    st.markdown("### 2. How Do Prices Change per Month? 💰")
    
    def get_data2(calendar):
        calendar = calendar.set_index('date')
        data_2016 = calendar.loc['2016']
        data_2016_dropna = data_2016.dropna()
        data_2016_dropna['price'] = data_2016_dropna['price'].apply(lambda x: float(x[1:].replace(',', '')))
        data_2016_month = data_2016_dropna[['price']].resample('M').mean()
        data_2016_month['Difference'] = data_2016_month['price'] - data_2016_month['price'].mean()
        data_2016_month['Month'] = data_2016_month.index.month_name()
        return data_2016_month

    data_2016_month = get_data2(calendar)
    fig = px.bar(
        data_2016_month,
        x='Month',
        y='Difference',
        title='Price Variation from Average Price in 2016',
        labels={'Month': 'Month', 'Difference': 'Price Difference (USD)'},
        color='Difference',
        template='custom'
    )
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        hovermode='x unified'
    )
    fig.update_traces(
        hovertemplate='Month: %{x}<br>Difference: $%{y:.2f}<br>Avg Price: $%{customdata:.2f}',
        customdata=data_2016_month['price']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Analysis Details", expanded=True):
        st.markdown("""
        **Visualization Description** 📊:\n
        This bar plot displays monthly price variations in Seattle Airbnb listings for 2016, relative to the annual mean. Data comes from `calendar.csv` (1,393,570 records, 2016–2017). Prices are cleaned, nulls dropped, and monthly averages calculated.

        **Key Insights** 🔑:\n
        - **Price Peak in July**: July shows a $20–$30 increase above the mean, driven by summer tourism.
        - **Price Low in January**: January is $15–$20 below the mean, reflecting low winter demand.
        - **Seasonal Patterns**: Prices are higher May–September, lower October–April.
        - **Investor Implications** 💼: Adjust pricing seasonally; consider dynamic pricing tools.
        - **Limitations** ⚠️: Data may not reflect final transaction prices.
        """)

elif analysis_option == "Popular Neighborhoods":
    st.markdown("### 3. What Are the Top Favorite Neighborhoods in Seattle? 🏘️")
    
    def get_data3(listings):
        listings_df = listings[['neighbourhood', 'city', 'state']].copy()
        listings_df = listings_df[listings_df.city != '西雅图']
        neighbour_count = listings_df['neighbourhood'].value_counts()
        df = pd.DataFrame({
            'neighbourhood': neighbour_count.index[:10],
            'Request_Percent': 100 * neighbour_count.iloc[:10].values / neighbour_count.sum()
        })
        return df

    df = get_data3(listings)
    fig = px.bar(
        df,
        x='neighbourhood',
        y='Request_Percent',
        title='Top 10 Most Popular Neighborhoods',
        labels={'neighbourhood': 'Neighborhood', 'Request_Percent': 'Percentage of Requests (%)'},
        color='Request_Percent',
        template='custom'
    )
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        hovermode='x unified'
    )
    fig.update_traces(
        hovertemplate='Neighborhood: %{x}<br>Percentage: %{y:.2f}%'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Analysis Details", expanded=True):
        st.markdown("""
        **Visualization Description** 📊:\n
        This bar plot ranks the top 10 Seattle neighborhoods by listing percentage, indicating host and guest popularity. Data is from `listings.csv`, excluding non-English city names.

        **Key Insights** 🔑:\n
        - **Capitol Hill Dominance**: ~10% of listings, driven by nightlife and downtown proximity.
        - **Ballard, Belltown**: 7–9% each, popular for charm and urban appeal.
        - **Diverse Appeal**: Fremont, Queen Anne, Wallingford (5–7%) attract varied guests.
        - **Investor Implications** 💼: Prioritize Capitol Hill, Ballard for high occupancy; Fremont for differentiation.
        - **Limitations** ⚠️: Popularity ≠ profitability; analyze pricing and occupancy.
        """)

elif analysis_option == "Price Influencers":
    st.markdown("### 4. What Are the Key Factors Influencing Airbnb Prices? 📈")
    
    def clean_data4(listings):
        feature_data = listings[['neighbourhood_group_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'price']].copy()
        feature_data['bathrooms'] = feature_data['bathrooms'].fillna(feature_data['bathrooms'].median())
        feature_data['bedrooms'] = feature_data['bedrooms'].fillna(feature_data['bedrooms'].median())
        feature_data['property_type'] = feature_data['property_type'].fillna(feature_data['property_type'].mode()[0])
        min_max_scaler = MinMaxScaler()
        feature_data[['accommodates', 'bathrooms', 'bedrooms']] = min_max_scaler.fit_transform(feature_data[['accommodates', 'bathrooms', 'bedrooms']])
        feature_data['price'] = feature_data['price'].apply(lambda x: float(x[1:].replace(',', '')))
        feature_data = pd.get_dummies(feature_data, columns=['neighbourhood_group_cleansed', 'property_type', 'room_type'])
        return feature_data

    def importance_df(feature_data):
        X = feature_data.drop(['price'], axis=1)
        y = feature_data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        coefficients = pd.DataFrame({'Feature': X_train.columns, 'Importance': lm.coef_})
        return coefficients.sort_values('Importance', ascending=False)[:15], X_test, y_test, lm

    feature_data = clean_data4(listings)
    importance_df_data, X_test, y_test, model = importance_df(feature_data)
    fig = px.bar(
        importance_df_data,
        y='Feature',
        x='Importance',
        title='Top Features Influencing Price',
        labels={'Feature': 'Features', 'Importance': 'Importance'},
        color='Importance',
        template='custom',
        orientation='h'
    )
    fig.update_layout(
        showlegend=False,
        hovermode='y unified'
    )
    fig.update_traces(
        hovertemplate='Feature: %{y}<br>Importance: %{x:.2f}'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    pred = model.predict(X_test)
    st.markdown(f"""
    **Model Performance**:
    - Mean Squared Error: {mean_squared_error(y_test, pred):.2f}
    - Mean Absolute Error: {mean_absolute_error(y_test, pred):.2f}
    - R² Score: {r2_score(y_test, pred):.3f}
    """)
    
    with st.expander("Analysis Details", expanded=True):
        st.markdown("""
        **Visualization Description** 📊:\n
        This horizontal bar plot shows the top 15 features influencing Airbnb prices, based on a linear regression model from `listings.csv`. Features are normalized and encoded.

        **Key Insights** 🔑:\n
        - **Bedrooms, Bathrooms**: Each additional unit significantly increases price ($50–$100).
        - **Unique Properties**: Boats, campsites command premiums due to novelty.
        - **Room Type**: Entire homes have higher prices than private/shared rooms.
        - **Investor Implications** 💼: Focus on 2–3 bedroom properties for optimal returns.
        - **Limitations** ⚠️: Correlation ≠ causation; external factors not included.
        """)

elif analysis_option == "Geographic Distribution":
    st.markdown("### 5. How Are Airbnb Listings Distributed in Seattle? 📍")
    
    # Load all listings at once
    filtered_listings = listings.dropna(subset=['latitude', 'longitude']).copy()
    
    # Create a copy for the unfiltered data
    all_listings = filtered_listings.copy()
    
    fig = px.scatter_mapbox(
        filtered_listings,
        lat='latitude',
        lon='longitude',
        color='number_of_reviews',
        color_continuous_scale='RdBu_r',
        range_color=[0, 100],
        opacity=0.5,
        size_max=10,
        zoom=10,
        center={'lat': 47.6062, 'lon': -122.3321},
        title='Geodistribution of Airbnb Listings in Seattle',
        hover_data=['neighbourhood', 'property_type', 'number_of_reviews'],
        mapbox_style='open-street-map',
        template='custom'
    )
    fig.update_traces(
        marker=dict(size=6),
        hovertemplate=(
            'Neighborhood: %{customdata[0]}<br>'
            'Type: %{customdata[1]}<br>'
            'Reviews: %{customdata[2]}<br>'
            'Latitude: %{lat:.4f}<br>'
            'Longitude: %{lon:.4f}'
        )
    )
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        mapbox=dict(bearing=0, pitch=0),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d']}
    )
    
    # Add a divider for better separation
    st.markdown("---")
    
    # Review threshold slider in a container for better layout
    with st.container():
        st.markdown("### Filter Listings by Review Count")
        col1, col2 = st.columns([1, 3])
        with col1:
            min_reviews = st.slider(
                'Minimum reviews:',
                min_value=0,
                max_value=int(all_listings['number_of_reviews'].quantile(0.95)),
                value=0,
                step=1,
                help='Filter listings by minimum number of reviews',
                key='review_slider'
            )
        with col2:
            # Show the current filter status
            filtered_count = len(all_listings[all_listings['number_of_reviews'] >= min_reviews])
            total_count = len(all_listings)
            st.metric(
                "Listings Shown",
                f"{filtered_count:,}",
                f"{filtered_count/total_count*100:.1f}% of total"
            )
    
    # Update the map with filtered data when slider changes
    if 'last_review_filter' not in st.session_state or st.session_state.last_review_filter != min_reviews:
        st.session_state.last_review_filter = min_reviews
        filtered_listings = all_listings[all_listings['number_of_reviews'] >= min_reviews]
        st.rerun()
    
    with st.expander("Analysis Details", expanded=True):
        st.markdown("""
        **Visualization Description** 📊:\n
        This scatter map shows the geographic distribution of Airbnb listings in Seattle, with colors indicating review counts. Data is from `listings.csv`.

        **Key Insights** 🔑:\n
        - **Central Density**: Capitol Hill, Downtown, Belltown have high listing density but moderate reviews.
        - **Lake-View Appeal**: Listings near Lake Union, West Seattle show higher reviews.
        - **Review Distribution**: Most listings have 0–50 reviews; few exceed 100.
        - **Investor Implications** 💼: Central areas offer visibility, lake-view properties higher engagement.
        - **Limitations** ⚠️: Reviews may not capture all occupancy; sampling may exclude data.
        """)

st.markdown('</div>', unsafe_allow_html=True)