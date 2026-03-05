import streamlit as st
import google.generativeai as genai
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="AI 課堂筆記助手", page_icon="🎓")
st.title("🎓 AI 課堂筆記助手")

# --- 從 Secrets 讀取 API Key ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("❌ 請在 Streamlit Secrets 中設定 GROQ_API_KEY 與 GEMINI_API_KEY")
    st.stop()

# 初始化客戶端
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --- 介面佈局 ---
tab1, tab2 = st.tabs(["🎤 現錄語音", "📂 上傳檔案"])
audio_source = None

with tab1:
    st.write("點擊下方圖示開始錄製：")
    # 【修正 1】確保括號完全閉合，並使用最精簡參數預防 TypeError
    audio_record = mic_recorder(
        start_prompt="🎤 開始錄音",
        stop_prompt="🛑 停止錄音",
        key='recorder'
    )
    
    if audio_record:
        audio_source = {"content": audio_record['bytes'], "name": "recorded_audio.mp3"}
        st.audio(audio_source["content"])
        st.success("✅ 錄音已就緒！")

with tab2:
    uploaded_file = st.file_uploader("選擇音檔 (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if uploaded_file:
        audio_source = {"content": uploaded_file.read(), "name": uploaded_file.name}
        st.audio(audio_source["content"])

# --- 處理邏輯 ---
if audio_source:
    if st.button("🚀 開始製作 AI 筆記"):
        try:
            # 1. Groq 轉錄
            with st.spinner("正在辨識語音中..."):
                transcription = groq_client.audio.transcriptions.create(
                    file=(audio_source["name"], audio_source["content"]),
                    model="whisper-large-v3",
                    response_format="text",
                    language="zh"
                )
                transcript_text = transcription
            
            # 2. Gemini 整理
            with st.spinner("正在生成筆記..."):
                # 【修正 2】解決 404 問題：改用 gemini-1.5-flash 並指定正確參數名
                # 若 gemini-1.5-flash 仍報錯，請將下面改為 'gemini-pro'
                model = genai.GenerativeModel(model_name='gemini-1.5-flash')
                
                prompt = f"你是一位專業的筆記秘書。請將以下逐字稿整理成結構化、條理分明的筆記：\n\n{transcript_text}"
                response = model.generate_content(prompt)
                ai_note = response.text

            # 3. 顯示與下載
            st.subheader("✨ AI 整理後的筆記")
            st.markdown(ai_note)
            
            st.divider()
            today = datetime.date.today().strftime("%Y%m%d")
            st.download_button(
                label="📥 下載筆記 (.md)",
                data=ai_note,
                file_name=f"課堂筆記_{today}.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"❌ 發生錯誤：{e}")
            st.info("提示：如果出現 404 錯誤，請檢查 requirements.txt 是否包含 google-generativeai>=0.5.0")
else:
    st.info("💡 請先錄音或上傳音檔。")
