import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

data=pd.read_csv("salary_data.csv")
x=np.array(data["YearsExperience"]).reshape(-1,1)
lr=LinearRegression()
lr.fit(x,np.array(data['Salary']))

st.title("Salery Predictor")

nav =st.sidebar.radio("Navigation",["home","prediction","contribute"])

if nav=="home":
    st.image("salery.jpg")
    if st.checkbox("show table"):
        st.table(data)

    graph=st.selectbox("What kind of graph u want ?",["Non-Interactive","Interactive"])
    val=st.slider("Filter data using years",0,20)
    data=data.loc[data["YearsExperience"]>=val]
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if graph=="Non-Interactive":
        plt.figure(figsize=(10,5))
        plt.scatter(data["YearsExperience"],data["Salary"])
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot()

    if graph=="Interactive":
        layout=go.Layout(
            xaxis=dict(range=[0,16]),
            yaxis=dict(range=[0,2100000])
        )
        fig=go.Figure(data=go.Scatter(x=data["YearsExperience"],y=data["Salary"],mode="markers"),layout = layout)
        st.plotly_chart(fig)
        


if nav=="prediction":
    st.header("Know your salary")
    val=st.number_input("Enter your exp",0.00,20.00,step=0.25)
    val=np.array(val).reshape(1,-1)
    pred=lr.predict(val)[0]
    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")


if nav=="contribute":
    st.header("Contribute to our dataset")
    ex=st.number_input("Enter your experience",0.00,20.0)
    sal=st.number_input("Enter your salary",0.00,10000000.00,step=1000.0)
    if st.button("Submit"):
        to_add = {"YearsExperience": [ex], "Salary": [sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("salary_data.csv", mode='a', header=False, index=False)
        st.success("Submitted. Thank You!")
     


