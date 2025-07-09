import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from scipy.sparse import hstack
from wordcloud import WordCloud

st.set_page_config(page_title="📊 Prediksi Popularitas TikTok", layout="wide")
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stPlotlyChart div[data-testid="stVerticalBlock"] {
        overflow-x: visible !important;
        overflow-y: visible !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/tiktok_metadata.csv")
    return df

def preprocess_data(df):
    df = df.dropna(subset=['description', 'plays', 'author', 'music'])

    le_author = LabelEncoder()
    df['author_encoded'] = le_author.fit_transform(df['author'])
    le_music = LabelEncoder()
    df['music_encoded'] = le_music.fit_transform(df['music'])

    df['description_length'] = df['description'].str.len()
    df['description_word_count'] = df['description'].str.split().str.len()
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df['hour'] = df['create_time'].dt.hour
    df['day_of_week'] = df['create_time'].dt.dayofweek
    df['day'] = df['day_of_week'].map({0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'})
    df['popular'] = (df['plays'] >= 1_000_000).astype(int)

    df['hashtags'] = df['hashtags'].fillna('')
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['hashtags'])

    df = df.drop(columns=['video_id', 'fetch_time', 'views', 'posted_time'])

    return df, tfidf_matrix, df['popular'], tfidf, le_author, le_music

def evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Presisi", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    st.subheader("Laporan Klasifikasi")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

    st.subheader("Matriks Kebingungan")
    fig = px.imshow(confusion_matrix(y_test, y_pred),
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Popular', 'Popular'],
                    y=['Not Popular', 'Popular'],
                    color_continuous_scale='Blues', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

def predict_content(model, tfidf, le_author, le_music, user_input):
    hashtag_vec = tfidf.transform([user_input['hashtags']])
    author_enc = le_author.transform([user_input['author']])[0] if user_input['author'] in le_author.classes_ else 0
    music_enc = le_music.transform([user_input['music']])[0] if user_input['music'] in le_music.classes_ else 0

    desc_len = len(user_input['description'])
    word_count = len(user_input['description'].split())

    features = np.array([
        user_input['likes'], user_input['comments'], user_input['shares'],
        desc_len, word_count, author_enc, music_enc,
        user_input['hour'], user_input['day_of_week']
    ]).reshape(1, -1)

    X_final = hstack([features, hashtag_vec])
    pred = model.predict(X_final)[0]
    return pred

def predict_batch(df_upload, model, tfidf, le_author, le_music):
    df_upload = df_upload.dropna(subset=['description', 'author', 'music'])

    if 'hour' not in df_upload.columns:
        df_upload['hour'] = 12  # default jam 12
    if 'day_of_week' not in df_upload.columns:
        df_upload['day_of_week'] = 0  # default Senin

    df_upload['description_length'] = df_upload['description'].str.len()
    df_upload['description_word_count'] = df_upload['description'].str.split().str.len()
    df_upload['hashtags'] = df_upload['hashtags'].fillna('')
    df_upload['hour'] = df_upload['hour'].fillna(12).astype(int)
    df_upload['day_of_week'] = df_upload['day_of_week'].fillna(0).astype(int)

    df_upload['author_encoded'] = df_upload['author'].apply(
        lambda x: le_author.transform([x])[0] if x in le_author.classes_ else 0)
    df_upload['music_encoded'] = df_upload['music'].apply(
        lambda x: le_music.transform([x])[0] if x in le_music.classes_ else 0)

    tfidf_matrix = tfidf.transform(df_upload['hashtags'])

    features = df_upload[['likes', 'comments', 'shares', 'description_length', 'description_word_count',
                          'author_encoded', 'music_encoded', 'hour', 'day_of_week']].values

    X_batch = hstack([features, tfidf_matrix])
    predictions = model.predict(X_batch)
    
    # Add the plays column to the results
    df_upload['plays'] = df_upload.get('plays', np.nan)

    df_upload['prediksi_popularitas'] = np.where(predictions == 1, 'Populer', 'Tidak Populer')
    
    return df_upload

def main():
    st.sidebar.title("📊 Dashboard Sistem")
    if st.sidebar.button("📈 EDA dan Visualisasi Data"):
        st.session_state.section = 'EDA'
    if st.sidebar.button("🧠 Model Evaluasi Konten"):
        st.session_state.section = 'Model'
    if st.sidebar.button("📁 Informasi Data TikTok"):
        st.session_state.section = 'Data'
    if st.sidebar.button("🔍 Uji Popularitas Konten"):
        st.session_state.section = 'Uji'

    st.sidebar.markdown("---")
    st.sidebar.write("🎬 Dashboard Popularitas TikTok")

    if 'section' not in st.session_state:
        st.session_state.section = 'EDA'

    df = load_data()
    if df.empty:
        st.warning("Gagal memuat data.")
        return

    df, tfidf_matrix, y, tfidf, le_author, le_music = preprocess_data(df)

    if st.session_state.section == 'EDA':
        st.header("1. Analisis Data Eksploratif")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "📈 Distribusi Popularitas", "📊 Boxplot Interaksi", "🕒 Waktu Unggah Konten",
            "📅 Heatmap Hari & Jam", "📏 Rata-rata Panjang Deskripsi", "📈 Rata-rata Interaksi Harian",
            "📉 Scatter Likes vs Plays", "🌐 Wordcloud Hashtag", "🔍 Korelasi Fitur Numerik",
            "🎵 Musik Terpopuler"
        ])

        with tab1:
            fig = px.histogram(df, x='popular', color='popular', 
                               title='Distribusi Popularitas Konten',
                               color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig)

        with tab2:
            for col in ['likes', 'shares', 'comments']:
                fig = px.box(df, x='popular', y=col, color='popular', title=f'{col.capitalize()} vs Popularitas')
                st.plotly_chart(fig)

        with tab3:
            hour_filter = st.slider("Pilih Jam Unggah", 0, 23, (0, 23))
            filtered_df = df[(df['hour'] >= hour_filter[0]) & (df['hour'] <= hour_filter[1])]
            fig1 = px.histogram(filtered_df, x='hour', nbins=24, title='Distribusi Waktu Unggah Konten (per Jam)')
            fig2 = px.histogram(filtered_df, x='hour', color='popular', barmode='overlay', title='Popularitas Berdasarkan Jam Unggah')
            hour_popularity = filtered_df.groupby('hour')['popular'].mean().reset_index()
            fig3 = px.line(hour_popularity, x='hour', y='popular', markers=True, title='Proporsi Konten Populer per Jam')
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            st.plotly_chart(fig3)

        with tab4:
            pivot = df.pivot_table(index='day_of_week', columns='hour', values='popular', aggfunc='mean')
            day_labels = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
            pivot.index = [day_labels[i] for i in pivot.index]
            fig = px.imshow(pivot, text_auto='.2f', color_continuous_scale='YlGnBu',
                            title='Heatmap Popularitas Berdasarkan Hari dan Jam Unggah')
            st.plotly_chart(fig)

        with tab5:
            fig = px.histogram(df, x='description_length', color='popular', 
                               title='Distribusi Panjang Deskripsi',
                               marginal='rug', barmode='overlay')
            st.plotly_chart(fig)

            avg_desc = df.groupby('popular')['description_length'].mean().reset_index()
            fig2 = px.bar(avg_desc, x='popular', y='description_length', color='popular',
                          title='Rata-rata Panjang Deskripsi')
            st.plotly_chart(fig2)

        with tab6:
            day_map = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
            df['day_of_week_str'] = df['day_of_week'].map(lambda x: day_map[int(x)])
            avg_interactions = df.groupby('day_of_week_str')[['likes', 'comments', 'shares']].mean().reset_index()
            fig = px.bar(avg_interactions, x='day_of_week_str', y=['likes', 'comments', 'shares'], barmode='group',
                         title='Rata-rata Interaksi per Hari', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig)

        with tab7:
            fig = px.scatter(df, x='likes', y='plays', color='popular', log_x=True, log_y=True,
                             title='Likes vs Plays (log scale)', opacity=0.6)
            st.plotly_chart(fig)

        with tab8:
            text = ' '.join(df['hashtags'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig = px.imshow(wordcloud, title='Wordcloud Hashtags')
            st.plotly_chart(fig)

        with tab9:
            corr_cols = ['likes', 'comments', 'shares', 'plays', 'description_length', 'description_word_count', 'hour']
            corr = df[corr_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu',
                            title='Korelasi Fitur Numerik')
            st.plotly_chart(fig)

        with tab10:
            top_music_popular = df[df['popular'] == 1]['music'].value_counts().head(10)
            top_music_not_popular = df[df['popular'] == 0]['music'].value_counts().head(10)
            music_compare = pd.DataFrame({
                'Populer': top_music_popular,
                'Tidak Populer': top_music_not_popular
            }).fillna(0)
            music_compare['Total'] = music_compare.sum(axis=1)
            top10 = music_compare.sort_values('Total', ascending=False).head(10)
            fig = px.bar(top10, x=top10.index, y=['Populer', 'Tidak Populer'], 
                          title='Perbandingan Musik Populer vs Tidak', barmode='stack')
            st.plotly_chart(fig)

    elif st.session_state.section == 'Model':
        st.header("2. Evaluasi Model Random Forest")
        model = joblib.load("models/rf_model.pkl")
        X = hstack([df[['likes', 'comments', 'shares', 'description_length', 'description_word_count',
                         'author_encoded', 'music_encoded', 'hour', 'day_of_week']], tfidf_matrix])
        evaluate_model(model, X, y)  # Pass y as an argument

    elif st.session_state.section == 'Data':
        st.header("3. Informasi Dataset TikTok")
        st.dataframe(df)
        with st.expander("Statistik Deskriptif"):
            st.dataframe(df.describe())

    elif st.session_state.section == 'Uji':
        st.header("4. Uji Prediksi Popularitas Konten")
        model = joblib.load("models/rf_model.pkl")
        tab1, tab2 = st.tabs(["📝 Input Manual", "📤 Upload File CSV"])

        with tab1:
            st.subheader("Masukkan Data Konten TikTok")
            col1, col2 = st.columns(2)
            with col1:
                likes = st.number_input("Jumlah Likes", min_value=0, value=100)
                comments = st.number_input("Jumlah Comments", min_value=0, value=10)
                shares = st.number_input("Jumlah Shares", min_value=0, value=5)
                description = st.text_area("Deskripsi Konten", value="Konten TikTok yang menarik")

            with col2:
                hashtags = st.text_input("Hashtags (dipisah spasi)", value="#fun #viral")
                author = st.selectbox("Nama Author", df['author'].unique())
                music = st.selectbox("Nama Musik", df['music'].unique())
                hour = st.slider("Jam Upload (0-23)", 0, 23, 12)
                day_map = {'Senin': 0, 'Selasa': 1, 'Rabu': 2, 'Kamis': 3, 'Jumat': 4, 'Sabtu': 5, 'Minggu': 6}
                day_name = st.selectbox("Hari Upload", list(day_map.keys()))
                day_of_week = day_map[day_name]

            if st.button("🔮 Uji Popularitas Konten"):
                user_input = {
                    'likes': likes,
                    'comments': comments,
                    'shares': shares,
                    'description': description,
                    'hashtags': hashtags,
                    'author': author,
                    'music': music,
                    'hour': hour,
                    'day_of_week': day_of_week
                }
                prediction = predict_content(model, tfidf, le_author, le_music, user_input)
                if prediction == 1:
                    st.success("🎉 Konten ini **Populer**!")
                else:
                    st.info("🙁 Konten ini **Tidak Populer**.")

        with tab2:
            st.subheader("Upload File CSV untuk Popularitas Konten")
            uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file)
                st.write("📄 Data yang Diupload:")
                st.dataframe(df_upload.head())

                if st.button("🚀 Uji Data Unggah File"):
                    hasil = predict_batch(df_upload, model, tfidf, le_author, le_music)
                    st.success("✅ Hasil Uji selesai.")
                    st.dataframe(hasil[['description', 'likes', 'comments', 'shares', 'plays','prediksi_popularitas']])
                    csv = hasil.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Hasil Uji Data", data=csv, file_name="hasil_uji.csv", mime='text/csv')

if __name__ == '__main__':
    main()
