import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Customer Behavior Analysis", layout="wide")

# Title and description
st.title("Customer Behavior Analysis")
st.markdown("""
This app analyzes customer behavior across demographics, purchasing patterns, and income levels to provide actionable insights.
Data source: [Kaggle Customer Purchases Behaviour Dataset](https://www.kaggle.com/datasets/sanyamgoyal401/customer-purchases-behaviour-dataset)
""")

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('customer_data.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'customer_data.csv' not found. Please ensure the file is in the project directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function for data preprocessing
def preprocess_data(df):
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    if '60+' not in df:
        st.warning('No data for 60+')
    # Create income levels
    df['income_level'] = pd.cut(df['income'], bins=3, labels=['Low', 'Medium', 'High'])
    # Map purchase frequency to scores
    frequency_score_map = {'rare': 1, 'occasional': 2, 'frequent': 3}
    df['purchase_frequency_score'] = df['purchase_frequency'].map(frequency_score_map)
    return df

# Load data
df = load_data()

if df is not None:
    # Preprocess data
    df = preprocess_data(df)

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    section = st.sidebar.selectbox("Choose a section", 
                                  ["Data Overview", "Demographic Analysis", "Behavioral Analysis", "Recommendations"])

    # Data Overview
    if section == "Data Overview":
        st.header("Data Overview")
        st.write("Dataset Shape:", df.shape)
        st.write("Data Types:")
        st.write(df.dtypes)
        st.write("Missing Values:")
        st.write(df.isna().sum())
        st.write("Descriptive Statistics:")
        st.write(df.describe())

    # Demographic Analysis
    elif section == "Demographic Analysis":
        st.header("Demographic Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Age Distribution
            fig_age = px.histogram(df, x='age', nbins=30, title="Age Distribution of Customers")
            st.plotly_chart(fig_age, use_container_width=True)

            # Gender Distribution
            fig_gender = px.histogram(df, x='gender', color='gender', title="Gender Distribution")
            st.plotly_chart(fig_gender, use_container_width=True)

        with col2:
            # Income Distribution
            fig_income = px.histogram(df, x='income', nbins=30, title="Income Distribution of Customers")
            st.plotly_chart(fig_income, use_container_width=True)

            # Income by Education
            fig_edu = px.box(df, x='education', y='income', color='education', 
                             title="Income Distribution by Education Level")
            st.plotly_chart(fig_edu, use_container_width=True)

    # Behavioral Analysis
    elif section == "Behavioral Analysis":
        st.header("Behavioral Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Purchase Amount by Purchase Frequency
            fig_purchase = px.box(df, x='purchase_frequency', y='purchase_amount', 
                                  color='purchase_frequency', title="Purchase Amount by Purchase Frequency")
            st.plotly_chart(fig_purchase, use_container_width=True)

            # # Purchase Frequency by Age Group
            # fig_freq_age = px.box(df, x='purchase_frequency_score', color='age_group', 
            #                           title="Purchase Frequency Score by Age Group")
            
            # st.plotly_chart(fig_freq_age, use_container_width=True)

        with col2:
            # Correlation Matrix
            st.write("Correlation Matrix of Numeric Features")
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            sns.heatmap(corr, cmap='cividis', annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

    # Recommendations
    elif section == "Recommendations":
        st.header("Recommendations")
        st.markdown("""
        - **Younger (18–30) customers**: Spend more and purchase frequently. Launch age-targeted loyalty programs or referral incentives.
        - **High-income customers**: Not always the most satisfied. Review pricing vs. product category value proposition.
        - **High-satisfaction customers**: Tend to buy fewer times but spend more. Consider a VIP membership with early access or bundled deals.
        """)

else:
    st.write("Please fix the data loading issue to proceed.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data Analysis by [Abdullokh Samadov](https://www.github.com/haqqulzoda)")