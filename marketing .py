import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_farmer_info():
    info = {
        "Ramphala": {
            "Growth in India": "Cultivated to a limited extent in West Bengal, Gujarat, Maharashtra, Karnataka, Tamil Nadu, and Kerala.",
            "Commercial & Industrial Use": "Pulp is processed into ice creams and beverages.",
            "Pharmaceutical & Other Use": "Leaves possess insecticidal properties. Traditionally used to treat ulcers, diarrhea, and dysentery."
        },
        "Lakshmanphala": {
            "Growth in India": "Cultivated on a small scale in the southern states, including Tamil Nadu, Kerala, Karnataka, Andhra Pradesh, and Maharashtra.",
            "Commercial & Industrial Use": "Fruit is highly valued for processing into beverages, ice creams, and jellies.",
            "Pharmaceutical & Other Use": "Contains bioactive compounds (acetogenins) which are of significant interest for potential anticancer drug development."
        },
        "Wood Apple": {
            "Growth in India": "Grows wild and is cultivated throughout India, particularly in Uttar Pradesh, Bihar, West Bengal, Maharashtra, and the dry tracts of Southern India.",
            "Commercial & Industrial Use": "Pulp is widely processed into beverages (sherbet), jams, jellies, and toffees. Gum from the bark is used as a substitute for gum arabic. The hard shell can be used for charcoal.",
            "Pharmaceutical & Other Use": "Unripe fruit is famously used in traditional medicine (Ayurveda) to treat diarrhea and dysentery. Possesses hepatoprotective (liver-protecting), antimicrobial, and antiviral properties."
        }
    }
    return info

def get_consumer_info():
    info = {
        "Ramphala": {
            "Health Benefits": """
            * Traditionally used to treat diarrhea, dysentery, and ulcers.
            * Possesses significant antioxidant, anti-inflammatory, and antimicrobial properties.
            """
        },
        "Lakshmanphala": {
            "Health Benefits": """
            * Rich in Vitamin C, which helps boost immunity.
            * High fiber content aids in digestion and maintains gastrointestinal health.
            * Known for anti-inflammatory and antioxidant properties that combat cellular damage.
            * Widely studied for its acetogenin compounds and their potential anticancer effects.
            """
        },
        "Wood Apple": {
            "Health Benefits": """
            * Highly valued in traditional medicine; ripe fruit acts as a laxative, while unripe fruit helps treat diarrhea and dysentery.
            * Manages digestive disorders and is known to have liver-protecting properties.
            * Exhibits antimicrobial and antiviral activities.
            """
        }
    }
    return info

def get_nutritional_data():
    data = {
        'Fruit': ['Ramphala', 'Lakshmanphala', 'Wood Apple'],
        'Energy (kcal)': [101, 66, 137],
        'Protein (g)': [1.7, 1.0, 1.8],
        'Fat (g)': [0.6, 0.67, 0.3],
        'Carbohydrate (g)': [25.2, 16.84, 31.8],
        'Fiber (g)': [2.4, 3.3, 2.9],
        'Calcium (mg)': [30, 14, 85],
        'Phosphorus (mg)': [21, 27, 50],
        'Iron (mg)': [0.71, 0.6, 0.7],
        'Vitamin C (mg)': [19.2, 20.6, 8.0]
    }
    return pd.DataFrame(data).set_index('Fruit')

st.title("📊 Marketing & Growth Insights")
st.markdown("Data sourced from 'Indian Fruit Research: Uses and Potential.pdf'")

farmer_data = get_farmer_info()
consumer_data = get_consumer_info()
nutri_df = get_nutritional_data()

tab_farmer, tab_consumer = st.tabs(["Farmer Insights", "Consumer Insights"])

with tab_farmer:
    st.header("🧑‍🌾 Insights for Cultivators")
    
    st.subheader("Cultivation Areas in India")
    growth_data = {fruit: [info["Growth in India"]] for fruit, info in farmer_data.items()}
    growth_df = pd.DataFrame(growth_data, index=["States"]).T
    st.table(growth_df)

    st.subheader("Commercial, Pharmaceutical & Other Uses")
    for fruit, info in farmer_data.items():
        st.markdown(f"**{fruit}**")
        st.markdown(f"**Commercial & Industrial:** {info['Commercial & Industrial Use']}")
        st.markdown(f"**Pharmaceutical & Other:** {info['Pharmaceutical & Other Use']}")
        st.markdown("---")

with tab_consumer:
    st.header("👩‍🍳 Insights for Consumers")
    
    st.subheader("Health Benefits")
    for fruit, info in consumer_data.items():
        st.markdown(f"**{fruit}**")
        st.markdown(info['Health Benefits'])
        st.markdown("---")

    st.header("📈 Nutritional Comparison (per 100g)")
    st.markdown("Data from Tables 1, 2, and 3 in the research PDF.")

    st.subheader("Macronutrients & Energy")
    macro_data = nutri_df[['Energy (kcal)', 'Protein (g)', 'Fat (g)', 'Carbohydrate (g)', 'Fiber (g)']]
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    macro_data.plot(kind='bar', ax=ax1, rot=0)
    ax1.set_title("Macronutrient & Energy Comparison (per 100g)")
    ax1.set_ylabel("Value")
    ax1.set_xlabel("Fruit")
    ax1.legend(title="Nutrient", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1, bbox_inches='tight')

    st.subheader("Key Mineral Comparison")
    mineral_data = nutri_df[['Calcium (mg)', 'Phosphorus (mg)', 'Iron (mg)']]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    mineral_data.plot(kind='bar', ax=ax2, rot=0)
    ax2.set_title("Key Mineral Comparison (per 100g)")
    ax2.set_ylabel("Value (mg)")
    ax2.set_xlabel("Fruit")
    ax2.legend(title="Mineral", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2, bbox_inches='tight')

    st.subheader("Vitamin C Comparison")
    vit_c_data = nutri_df[['Vitamin C (mg)']]
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    vit_c_data.plot(kind='bar', ax=ax3, color='orange', legend=False, rot=0)
    ax3.set_title("Vitamin C Comparison (per 100g)")
    ax3.set_ylabel("Value (mg)")
    ax3.set_xlabel("Fruit")
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig3, bbox_inches='tight')


