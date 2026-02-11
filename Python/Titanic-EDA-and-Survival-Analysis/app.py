import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# PAGE CONFIGURATIONS:

st.set_page_config(page_title = "Titanic EDA Dashboard",layout = "wide")

st.title("🚢 Titanic Survival Analysis")
st.write("Exploratory Data Analysis Dashboard(EDA)")


# LOADING DATA :

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "train.csv")

df = pd.read_csv(data_path)

#  BASIC INFORMATION: 

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Passengers", df.shape[0])
col2.metric("Features", df.shape[1])
col3.metric("Survival Rate", f"{df['Survived'].mean()*100:.2f}%")

#  DATA PREVIEW - TOP 10 ROWS:

st.subheader("Dataset Preview")
st.dataframe(df.head(10))


#  MISSING VALUES :

st.subheader("Missing Values")

missing = df.isnull().sum()

fig0, ax0 = plt.subplots()

missing.plot(kind="bar", ax=ax0)

ax0.set_title("Missing Values per Column")

st.pyplot(fig0)


# SURVIVAL COUNT :

st.subheader("Overall Survival")

fig1, ax1 = plt.subplots()

sns.countplot(x = "Survived",data = df)

ax1.set_xticklabels(["Not Survived", "Survived"])

st.pyplot(fig1)


#  SURVIVAL BY GENDER :

st.subheader("Survival by Gender")

fig2, ax2 = plt.subplots()

sns.countplot( x = "Sex", hue = "Survived", data = df, ax = ax2 )

st.pyplot(fig2)


# ---------- SURVIVAL BY CLASS ----------

st.subheader("Survival by Passenger Class")

fig3, ax3 = plt.subplots()

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df,
    ax=ax3
)

st.pyplot(fig3)


# AGE DISTRIBUTION :

st.subheader("Age Distribution")

fig4, ax4 = plt.subplots()

sns.histplot( df["Age"] , bins = 30, kde = True, ax = ax4)

st.pyplot(fig4)


#  FARE DISTRIBUTION :

st.subheader("Fare Distribution")

fig5, ax5 = plt.subplots()

sns.histplot( df["Fare"], bins=30, kde=True, ax = ax5 )

st.pyplot(fig5)


#  CORRELATION HEATMAP :

st.subheader("Correlation Heatmap")

corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()

fig6, ax6 = plt.subplots(figsize=(6, 5))

sns.heatmap( corr, annot=True, cmap="coolwarm", ax = ax6 )

st.pyplot(fig6)


# INTERACTIVE FILTER On Passenger Class & Gender:

st.subheader("Survival Analysis by Class & Gender")

selected_class = st.selectbox( "Select Passenger Class", sorted(df["Pclass"].unique()) )

selected_gender = st.selectbox( "Select Gender", df["Sex"].unique() )

filtered = df[ (df["Pclass"] == selected_class) & (df["Sex"] == selected_gender) ]

survival_rate = filtered["Survived"].mean() * 100

st.info(
    f"Survival Rate: {survival_rate:.2f}% "
    f"({len(filtered)} passengers)"
)
