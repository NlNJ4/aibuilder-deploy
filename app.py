import streamlit as st
from algorithm import findmenu
import pandas as pd
from datetime import datetime
import os

st.title("แนะนำเมนูของหวาน")

st.write("สวัสดีครับ ผมเป็นนักศึกษาที่กำลังทำโครงงานในด้านของการแนะนำเมนูของหวาน")
st.write("ส่วนนี้ก็เป็นตัวของ Prototype ของผมเอง อาจจะมีผิดพลาดและเมนูไม่ครอบคลุมขนาดนั้น")
st.write("อาจจะลองค้นหาว่า เมนูของหวานที่เป็นเค้ก")

dt = st.text_input("INPUT HERE")

if st.button("Find Menu"):
    if dt:
        recommendations = findmenu(dt)
        if recommendations:
            for rec in recommendations:
                st.write(rec)
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

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Append the score to a CSV file
    feedback_data = {
        'Timestamp': [timestamp],
        'Satisfaction': [q1],
    }
    feedback_df = pd.DataFrame(feedback_data)
    if os.path.exists('feedback.csv'):
        feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
    else:
        feedback_df.to_csv('feedback.csv', mode='w', header=True, index=False)    
    st.success("Thank you for your feedback")