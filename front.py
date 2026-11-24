import streamlit as st
import requests

st.title('Hand checker')

st.text('upload image for checking')

file = st.file_uploader('Upload image',
                 type=['png', 'jpg', 'jpeg'])

if file and st.button('Predict'):
    st.image(file)
    response = requests.post('http://127.0.0.1:8000/predict', files={'image': file})
    if response.ok:
        st.success(f'result: {response.json()['Answer'].title()}')
    else:
        st.error('server error')