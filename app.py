import streamlit as st
import google.generativeai as genai
from groq import Groq
import os
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="AI 課堂筆記助手", page_icon="🎓")
st.title("🎓 AI 課堂筆記助手")
st.write("上傳課堂錄音，透過 Groq (Whisper) 與 Gemini 快速生成專業筆記。")

# --- 從 Secrets 讀取 API Key ---
# 請確保在 Streamlit Cloud 的 Settings -> Secrets 設定好這兩個 Key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("請在 Streamlit Secrets 中設定 GROQ_API_KEY 與 GEMINI_API_KEY")
    st.stop()

# 初始化客戶端
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --- 1. 檔案上傳 ---
uploaded_file = st.file_uploader("選擇音檔 (mp3, wav, m4a, mpeg)", type=["mp3", "wav", "m4a", "mpeg"])

if uploaded_file:
    st.audio(uploaded_file) # 讓使用者可以先試聽
    
    # 建立一個按鈕觸發後續流程
    if st.button("🚀 開始製作 AI 筆記"):
        
        try:
            # --- 2. 語音轉文字 (Groq Whisper) ---
            with st.spinner("正在使用 Groq 進行高速轉錄..."):
                # 將上傳的檔案讀取為位元組，並傳送給 Groq
                file_content = uploaded_file.read()
                
                # 注意：Groq 接收的是檔案元組 (檔名, 內容)
                transcription = groq_client.audio.transcriptions.create(
                    file=(uploaded_file.name, file_content),
                    model="whisper-large-v3", # 目前最強的開源模型
                    response_format="text",
                    language="zh" # 強制指定中文
                )
                transcript_text = transcription
            
            st.success("✅ 轉錄完成！")
            with st.expander("查看原始逐字稿"):
                st.write(transcript_text)

            # --- 3. AI 整理筆記 (Gemini) ---
            with st.spinner("正在使用 Gemini 整理重點..."):
                model = genai.GenerativeModel('gemini-1.5-flash') # 使用最新的 flash 模型，速度快且免費額度多
                prompt = f"""
                你是一位專業的課堂筆記秘書。請根據以下逐字稿內容，整理出一份條理清晰的筆記。
                要求包含：
                1. 課堂主題摘要
                2. 重點條列說明
                3. 關鍵名詞解釋
                4. 待辦事項或作業 (若有提到)
                
                逐字稿內容：
                {transcript_text}
                """
                response = model.generate_content(prompt)
                ai_note = response.text

            # --- 4. 顯示與下載結果 ---
            st.subheader("✨ AI 整理後的筆記")
            st.markdown(ai_note)

            st.divider()
            
            # 準備下載
            today = datetime.date.today().strftime("%Y%m%d")
            st.download_button(
                label="📥 下載筆記 (.md)",
                data=ai_note,
                file_name=f"課堂筆記_{today}.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"發生錯誤：{e}")

else:
    st.info("請先上傳一個錄音檔開始。")
