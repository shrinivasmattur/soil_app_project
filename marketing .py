import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def get_farmer_info():
    info = {
        "Ramphala": {
            "Market Price (India)": "₹9917 per Quintal",
            "Agricultural Yield": "N/A",
            "Export Status": "Niche market, less data on exports.",
            "Commercial Value": """
* **Pharmaceutical:** Valued for medicinal properties and high antioxidant content.
* **Nutraceutical:** Used in health-focused food products.
* **Environmental:** Waste (leaves/bark) is repurposed as a bio-adsorbent to remove industrial dyes.
"""
        },
        "Lakshmanphala": {
            "Market Price (India)": "Market in transformation, driven by health awareness. Specific price not listed.",
            "Agricultural Yield": "N/A",
            "Export Status": "Growing awareness is increasing its market presence.",
            "Commercial Value": """
* **Pharmaceutical:** Valued for immune-boosting and potential anti-cancer properties.
* **Nutraceutical:** High demand for use in health and wellness products.
* **Beverage Industry:** Popular for its distinct aromatic qualities in juices and teas.
"""
        },
        "Wood Apple": {
            "Market Price (India)": "Historically ~Rs 3 per piece (raw, local markets).",
            "Agricultural Yield": "40.50 to 70.00 kg per plant",
            "Export Status": "Internationally traded. China, South Korea, and USA are top exporters.",
            "Commercial Value": """
* **Food Processing:** Extensively used for value-added products like jams, jellies, beverages, and pickles.
* **Pharmaceutical:** Bioactive compounds used in anti-cancer and anti-diabetic formulations.
* **Cosmetics:** Extracts used for antioxidant properties to detoxify skin and strengthen hair.
* **Waste Valorization:** Shells are used as a highly effective bio-adsorbent for industrial dyes and as a reinforcing filler in composite materials.
"""
        }
    }
    return info

def get_consumer_info():
    info = {
        "Ramphala": {
            "Primary Consumption": "Primarily consumed fresh.",
            "Target Demographic": "Educated consumers, aged 31-45, who prioritize health, taste, and appearance.",
            "Health & Daily Life": """
* **Health:** High in antioxidants and valued for medicinal properties.
* **Driver:** Demand is driven by its unique flavor, vibrant color, and nutritional profile.
"""
        },
        "Lakshmanphala": {
            "Primary Consumption": "Consumed fresh and as a beverage (e.g., soursop tea).",
            "Target Demographic": "Educated consumers, aged 31-45. Demand is growing rapidly due to health awareness.",
            "Health & Daily Life": """
* **Health:** Known for immune-boosting and other health benefits.
* **Driver:** Strong awareness of its medicinal properties and unique aromatic quality.
"""
        },
        "Wood Apple": {
            "Primary Consumption": "Primarily consumed in processed forms (beverages, jams, pickles).",
            "Target Demographic": "Wide acceptability for its traditional use and in value-added products.",
            "Health & Daily Life": """
* **Health:** Known as a "miracle fruit" with anti-diabetic, anti-inflammatory, and antimicrobial properties.
* **Driver:** High consumer acceptability of processed products; beverages have a shelf-life of up to 50 days.
"""
        }
    }
    return info

st.title("📊 Marketing & Growth Insights")
st.markdown("Visual data to promote the cultivation and consumption of indigenous fruits, based on market research.")

tab_farmer, tab_consumer = st.tabs(["🧑‍🌾 Farmer Insights (Promotion)", "👩‍🍳 Consumer Insights (Promotion)"])

with tab_farmer:
    st.header("Promoting Growth: The Farmer's Perspective")
    farmer_data = get_farmer_info()
    farmer_df = pd.DataFrame(farmer_data).T
    
    st.markdown("These crops offer diverse opportunities in both local and international markets, from food processing to pharmaceuticals.")
    
    st.subheader("Key Agricultural & Market Statistics")
    st.table(farmer_df[['Market Price (India)', 'Agricultural Yield', 'Export Status']])

    st.subheader("Key Commercial & Industrial Uses")
    for fruit in farmer_df.index:
        st.markdown(f"**{fruit}**")
        st.markdown(farmer_df.loc[fruit, "Commercial Value"])
    
    st.markdown("---")
    st.subheader("Data-Driven Charts (from PDF)")
    st.markdown("Note: The charts below are based on the specific quantitative data available in the research document.")

    st.markdown("#### Wood Apple: International Export Leaders (by Shipments)")
    export_data = {
        'China': 2098,
        'South Korea': 956,
        'USA': 65
    }
    fig1, ax1 = plt.subplots()
    ax1.bar(export_data.keys(), export_data.values(), color=['#FF5733', '#C70039', '#900C3F'])
    ax1.set_ylabel('Number of Shipments')
    ax1.set_title('Top Wood Apple Exporters (from PDF data)')
    st.pyplot(fig1)

    st.markdown("#### Wood Apple: Agricultural Yield per Plant (India)")
    yield_data = {
        'Minimum Yield': 40.50,
        'Maximum Yield': 70.00
    }
    fig2, ax2 = plt.subplots()
    ax2.bar(yield_data.keys(), yield_data.values(), color=['#4CAF50', '#8BC34A'])
    ax2.set_ylabel('Yield (kg per plant)')
    ax2.set_title('Wood Apple Yield Range (from PDF data)')
    st.pyplot(fig2)


with tab_consumer:
    st.header("Promoting Consumption: The Consumer's Perspective")
    consumer_data = get_consumer_info()
    consumer_df = pd.DataFrame(consumer_data).T
    
    st.markdown("These fruits are gaining popularity among health-conscious consumers and are available in both fresh and processed forms.")
    
    st.subheader("Health Benefits & Consumption Patterns")
    st.table(consumer_df[['Primary Consumption', 'Target Demographic']])

    st.subheader("Key Promotional Points for Consumers")
    for fruit in consumer_df.index:
        st.markdown(f"**{fruit}**")
        st.markdown(consumer_df.loc[fruit, "Health & Daily Life"])