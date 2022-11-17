import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json



with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#태그출력
# st.markdown('<h1>ff</h1>', unsafe_allow_html=True)

#이미지출력
#st.image('./images/baby1.jpeg')






@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df
##
#
model = cached_model()
df = get_dataset()
st.markdown('[Main](https://www.youtube.com/c/빵형의개발도상국) | [Psychotherapy](https://www.youtube.com/c/빵형의개발도상국) | [Game](https://www.youtube.com/c/빵형의개발도상국)')
#st.header('그리니의 심리상담소입니다.')
#st.markdown("[<](https://www.youtube.com/c/빵형의개발도상국)")
#st.header('[Main](https://www.youtube.com/c/빵형의개발도상국) 그리니의 심리상담소입니다.')
#st.markdown("[Main](https://www.youtube.com/c/빵형의개발도상국)")



if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# clear_on_submit=True 말하기버튼누르면 입력란리셋
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('Greeni Psychotherapy')
    submitted = st.form_submit_button('Enter')
   # submitted2 = st.form_submit_button('말하기2')
    

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

chatlog = []
for i in range(len(st.session_state['past'])):
    chatlog.append(i)
chatlogReversed = list(reversed(chatlog))
for i in chatlogReversed:
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
 