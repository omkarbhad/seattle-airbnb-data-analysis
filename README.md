# 🏠 Seattle Airbnb Data Analysis Dashboard

An interactive dashboard for analyzing Seattle's Airbnb market trends, pricing factors, and neighborhood insights. Built with Python and Streamlit.

![Demo](demo.gif)

## ✨ Features

- **📊 Interactive Visualizations**
  - Seasonal occupancy trends
  - Price analysis across neighborhoods
  - Geographic distribution mapping

- **📈 Business Insights**
  - Peak season identification (June-August, 10-15% higher prices)
  - Top performing neighborhoods (Capitol Hill, Downtown, Belltown)
  - Price influencers analysis

- **🛠️ Technical Implementation**
  - Data preprocessing with Pandas
  - Machine learning with scikit-learn
  - Interactive visualizations using Plotly
  - Web application with Streamlit

## ❓ Key Questions Answered

1. **Seasonal Trends**
   - When are the peak and off-peak seasons for Airbnb bookings in Seattle?
   - How do prices fluctuate throughout the year?

2. **Pricing Analysis**
   - What factors most significantly impact listing prices?
   - How do prices vary across different neighborhoods?

3. **Neighborhood Performance**
   - Which neighborhoods have the highest number of listings?
   - Which areas offer the best value for money?

4. **Property Features**
   - How do property characteristics affect pricing and occupancy?
   - What amenities are most valued by guests?

5. **Geographic Distribution**
   - Where are the highest concentration of Airbnbs located?
   - How does location affect pricing and demand?

## 🚀 Quick Start

1. **Prerequisites**
   - Python 3.8+
   - pip

2. **Installation**
   ```bash
   # Clone the repository
   git clone https://github.com/your-username/seattle-airbnb-analysis.git
   cd seattle-airbnb-analysis
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```
   The app will open in your default web browser at `http://localhost:8501`

## 📊 Data Sources

- [Airbnb Seattle Data on Kaggle](https://www.kaggle.com/airbnb/seattle)
- Includes three main datasets:
  - `listings.csv`: Detailed information about listings
  - `reviews.csv`: Review data including dates and comments
  - `calendar.csv`: Pricing and availability data

## 🧠 Methodology

1. **Data Preparation**
   - Cleaned and preprocessed raw data
   - Handled missing values
   - Performed feature engineering

2. **Analysis**
   - Explored seasonal trends
   - Analyzed neighborhood performance
   - Identified price influencers

3. **Modeling**
   - Linear regression for price prediction
   - Feature importance analysis

## 📖 Blog Post

For a detailed walkthrough of the analysis and insights, check out my [Medium article](https://medium.com/@omkarbhad/seattle-airbnbs-data-analysis-that-every-investor-must-know-8a25a694389e).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
