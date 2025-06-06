import streamlit as st
import pandas as pd
import numpy as np

# Writing Title
st.title("Erste Streamlit-Anwendung")

st.write("""
In dieser ersten Version verwenden wir lediglich einen längeren
Text (String), ohne eine eigentliche Anweisung zu verweden. Während
die Anwendung läuft, kann jederzeit Code angepasst werden.
""")

st.subheader("LaTeX Code")
st.latex(r'''cos2\theta = 1 - 2sin^2\theta''')
st.latex("""(a+b)^2 = a^2 + b^2 + 2ab""")

# Displaying Python Code
st.subheader("Python Code")
code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language='python')

# Displaying Java Code
st.subheader("""Java Code""")
st.code("""public class MyClass {
    public static void main(String args[]) {
        System.out.println("Hello World");
     }
}""", language='java')

st.subheader("Data Frames")
# defining random values in a dataframe using pandas and numpy
df = pd.DataFrame(
    np.random.randn(30, 10), columns=('col %d' % i for i in range(10)))
st.dataframe(df)

st.subheader('Metriken')
st.metric(label="Temperatur", value="31 °C", delta="1.5 °C")
