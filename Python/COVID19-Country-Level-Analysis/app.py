import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# PAGE CONFIG :

st.set_page_config(page_title="COVID-19 Country Analysis", layout="wide")

st.title("🦠 COVID-19 Country Level Analysis")
st.write("Exploratory Data Analysis Dashboard")


# LOAD DATA :

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "country_wise_latest.csv")

df = pd.read_csv(data_path)


# SIDEBAR for Number of Countries Selection:

st.sidebar.header("Options")

top_n = st.sidebar.slider("Top N Countries", 5, 20, 10)


#  DATA PREVIEW TOP 5:

st.subheader("Dataset Preview")
st.dataframe(df.head(5))


# TOP N CONFIRMED CASES :

st.subheader("Top Countries by Confirmed Cases")

top_cases = df.sort_values("Confirmed", ascending=False).head(top_n)

fig1, ax1 = plt.subplots()

sns.barplot(
    x = "Confirmed",
    y = "Country/Region",
    data = top_cases,
    ax = ax1
)

st.pyplot(fig1)


# TOP N DEATHS :

st.subheader("Top Countries by Deaths")

top_deaths = df.sort_values("Deaths", ascending=False).head(top_n)

fig2, ax2 = plt.subplots()

sns.barplot(
    x = "Deaths",
    y = "Country/Region",
    data = top_deaths,
    ax = ax2
)

st.pyplot(fig2)

# TOP N Recoverd Countries :

st.subheader("Top Countries by Recoverd")

top_recovered = df.sort_values('Recovered', ascending=False).head(top_n)

fig7, ax7 = plt.subplots()

sns.barplot(
    x = "Recovered",
    y = "Country/Region",
    data = top_recovered,
    ax = ax7
)

st.pyplot(fig7)

# TOP N Countries With Highest Mortality Rate:

st.subheader("Top Countries by Mortality Rate")

top_mortality = df.sort_values('Deaths / 100 Cases', ascending=False).head(top_n)

fig8, ax8 = plt.subplots()

sns.barplot(
    x = "Deaths / 100 Cases",
    y = "Country/Region",
    data = top_mortality,
    ax = ax8
)

st.pyplot(fig8)

# TOP N Countries by Weekly Growth:

st.subheader("Top Countries by Weekly Growth")

weekly_growth = df.sort_values('1 week % increase', ascending=False).head(top_n) 

fig9, ax9 = plt.subplots()

sns.barplot(
    x = "1 week % increase",
    y = "Country/Region",
    data = weekly_growth,
    ax = ax9
)

st.pyplot(fig9)


# CORRELATION MAP:

st.subheader("Correlation Between COVID Metrics")

fig3, ax3 = plt.subplots(figsize=(6,5))

sns.heatmap(
    df[["Confirmed", "Deaths", "Recovered", "Active"]].corr(),
    annot = True,
    cmap = "coolwarm",
    ax = ax3
)

st.pyplot(fig3)


# CONFIRMED vs DEATHS :

st.subheader("Confirmed vs Deaths Relationship")

fig4 = sns.lmplot(
    x = "Confirmed",
    y = "Deaths",
    data = df,
    height = 5
)

st.pyplot(fig4)


#  WHO REGION ANALYSIS :

st.subheader("WHO Region Summary")

region_data = df.groupby("WHO Region")[["Confirmed","Recovered","Deaths"]].sum()

region_data["RecoveryRate"] = (
    region_data["Recovered"] / region_data["Confirmed"] * 100
)

st.dataframe(region_data)


fig5, ax5 = plt.subplots()

sns.barplot(
    x = region_data.index,
    y = region_data["Confirmed"],
    ax = ax5
)

ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30)

st.pyplot(fig5)

