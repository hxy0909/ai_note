import streamlit as st
import whisper
import google.generativeai as genai
import os
import datetime

# 設定 Gemini API
genai.configure(api_key="AIzaSyANkRNrbdeGOcpicfhkwwuWLqqulwxcKD8")

st.title("🎓 AI 課堂筆記助手")
st.write("上傳課堂錄音檔，自動為你生成精簡筆記！")

# 1. 檔案上傳
uploaded_file = st.file_uploader("選擇音檔 (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 儲存暫存檔
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("音檔上傳成功！準備開始轉錄...")

    # 2. 語音轉文字 (使用 Whisper)
    if st.button("開始製作筆記"):
        with st.spinner("正在辨識語音中... (這可能需要幾分鐘)"):
            model = whisper.load_model("base") # 初學者建議用 base 模型速度較快
            result = model.transcribe("temp_audio.mp3")
            transcript = result["text"]
            
        st.subheader("原文逐字稿")
        st.write(transcript)

        # 3. 使用 AI 整理筆記
        with st.spinner("正在整理筆記..."):
            ai_model = genai.GenerativeModel('gemini-pro')
            prompt = f"請根據以下課堂逐字稿，整理成一份條理分明的筆記，包含重點摘要與待辦事項：\n\n{transcript}"
            response = ai_model.generate_content(prompt)
            
        st.subheader("✨ AI 整理後的筆記")

        st.markdown(response.text)

if response.text:
    st.subheader("✨ AI 整理後的筆記")
    note_content = response.text
    st.markdown(note_content)

    st.divider()
    
    # 讓使用者輸入想要存檔的名字
    today = datetime.date.today().strftime("%Y%m%d")
    custom_name = st.text_input("輸入存檔名稱：", value=f"課堂筆記_{today}")

    st.download_button(
        label="📥 下載筆記檔案",
        data=note_content,
        file_name=f"{custom_name}.md",
        mime="text/markdown",
    )
