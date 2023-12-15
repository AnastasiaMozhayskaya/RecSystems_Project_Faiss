import streamlit as st
import pandas as pd
import numpy as np
import ast
import faiss
from data.func import filter_by_ganre, embed_user

"""
## Сервис умного поиска сериалов 📽️
"""

df = pd.read_csv('data/dataset.csv')
embeddings = np.load('data/embeddings_main.npy')
index = faiss.read_index('data/faiss_index_main.index')

df['ganres'] = df['ganres'].apply(lambda x: ast.literal_eval(x))

st.write(f'<p style="font-family: Arial, sans-serif; font-size: 24px; ">Наш сервис насчитывает \
         {len(df)} лучших сериалов</p>', unsafe_allow_html=True)

st.image('images/ser2.png')

ganres_lst = sorted(['драма', 'документальный', 'биография', 'комедия', 'фэнтези', 'приключения', 'для детей', 'мультсериалы', 
              'мелодрама', 'боевик', 'детектив', 'фантастика', 'триллер', 'семейный', 'криминал', 'исторический', 'музыкальные', 
              'мистика', 'аниме', 'ужасы', 'спорт', 'скетч-шоу', 'военный', 'для взрослых', 'вестерн'])

st.sidebar.header('Панель инструментов :gear:')
choice_g = st.sidebar.multiselect("Выберите жанры", options=ganres_lst)
n = st.sidebar.selectbox("Количество отображаемых элементов на странице", options=[5, 10, 15, 20, 30])
st.sidebar.info("📚 Для наилучшего соответствия, запрос должен быть максимально развернутым")

text = st.text_input('Введите описание для рекомендации')

button = st.button('Отправить запрос', type="primary")
    
if text and button:
    if len(choice_g) == 0:
        choice_g = ganres_lst
    filt_ind = filter_by_ganre(df, choice_g)
    user_emb = embed_user(filt_ind, embeddings, text)
   
    D, sorted_indices = index.search(user_emb.reshape(1, -1), n)
    top_ind = list(sorted_indices[0])
    output_dict = {}
    for i in top_ind:
        for ganre in df['ganres'][i]:
            if ganre in choice_g:
                output_dict[i] = df['ganres'][i]
    # st.write('output_dict')
    sorted_lst = sorted(output_dict.items(), key=lambda x: len(set(x[1]) & set(choice_g)), reverse=True)
    n_lst = [i[0] for i in sorted_lst[:n]]

    st.write(f'<p style="font-family: Arial, sans-serif; font-size: 18px; text-align: center;"><strong>Всего подобранных \
         рекомендаций {len(sorted_lst)}</strong></p>', unsafe_allow_html=True)
    st.write('\n')

    # Отображение изображений и названий
    for i in n_lst:
        col1, col2 = st.columns([3, 4]) 
        with col1:
            st.image(df['poster'][i], width=300)
        with col2:
            st.write(f"***Название:*** {df['title'][i]}")
            st.write(f"***Жанр:*** {', '.join(df['ganres'][i])}")
            st.write(f"***Описание:*** {df['description'][i]}")
            st.markdown(f"[***ссылка на сериал***]({df['url'][i]})")
            st.write("")
            # st.write(f"<small>*Степень соответствия по косинусному сходству: {conf_dict[i]:.4f}*</small>", unsafe_allow_html=True)
        st.markdown(
        "<hr style='border: 2px solid #000; margin-top: 10px; margin-bottom: 10px;'>",
        unsafe_allow_html=True
    )