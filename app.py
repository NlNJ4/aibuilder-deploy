import streamlit as st
from algorithm import findmenu
import pandas as pd
from datetime import datetime
import gspread
from searchimage import google_image_search,display_images
from oauth2client.service_account import ServiceAccountCredentials

# Connect to Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("collectscore.json", scope)
client = gspread.authorize(creds)
Score = client.open("Score").worksheet('Score')

# Title
st.title("แนะนำเมนูของหวาน")

# Introduction text
st.write("สวัสดีครับ ผมเป็นนักศึกษาที่กำลังทำโครงงานในด้านของการแนะนำเมนูของหวาน")
st.write("ส่วนนี้ก็เป็นตัวของ Prototype ของผมเอง อาจจะมีผิดพลาดและเมนูไม่ครอบคลุมขนาดนั้นนะครับ")
st.write("ถ้าลองใช้งานเสร็จแล้วอยากรบกวนให้คะแนนความถึงพอใจด้านล่างให้ผมเพื่อการปรับปรุงและพัฒนาต่อไปด้วยนะครับ")
st.write("วิธีการใช้คืออาจจะลองค้นหาว่า เมนูของหวานที่เป็นเค้ก")

dt = st.text_input("INPUT HERE")

if st.button("Find Menu"):
    if dt:
        recommendations = findmenu(dt)
        if recommendations:
            for rec in recommendations:
                st.write(rec)
                url = google_image_search(rec+" dessert")
                display_images(url)
        else:
            st.write("No recommendations found.")
    else:
        st.write("Please enter a valid input.")

# Footer section for the questionnaire
st.markdown("---")
st.subheader("Feedback")

with st.form("แบบสอบถาม"):
    st.write("พึงพอใจกับการแนะนำของเรามากแค่ไหนครับ")

    q1 = st.slider("ให้คะแนนความพึงพอใจ 0 - 5 คะแนน", 0, 5)
    
    submitted = st.form_submit_button("Submit")

if submitted:
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Append the score to Google Sheets
        new_row = [timestamp, q1]
        Score.append_row(new_row)

        st.success("Thank you for your feedback")
    except Exception as e:
        st.write("An error occurred while saving your feedback.")
        st.error(f"Error: {e}")