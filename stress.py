import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Academic Stress Detector", layout="centered", page_icon="📘")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", [
    "📘 Overview",
    "📝 Text Analysis",
    "📋 Survey Prediction",
    "📊 Visualization",
    "📆 Weekly Mood Tracker",
    "📬 Calendar Reminder",
    "💬 Peer Support Wall",
    "🧠 Chat Assistant",
    "🧩 Mind Games",
    "🎵 Sound Therapy",
    "📓 Mood Journal",
    "🧪 Coping Quiz",
    "💾 Export Predictions"
])

st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💻 Built with ❤️ by Shivangi Singh")

if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

dataset_path = "C:\\Users\\shivangi\\OneDrive\\Desktop\\student prediction\\MentalHealthSurvey.csv"
model = None
if os.path.exists(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        cols = ['academic_pressure', 'academic_workload', 'anxiety', 'depression']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=cols)
        df['stress_score'] = df[cols].mean(axis=1)

        def label_stress(score):
            if score < 3:
                return 'Low'
            elif score < 6:
                return 'Medium'
            else:
                return 'High'

        df['stress_level'] = df['stress_score'].apply(label_stress)
        X = df[cols]
        y = df['stress_level']
        model = RandomForestClassifier()
        model.fit(X, y)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.warning("Dataset not found. Please ensure the CSV file is available.")

def get_ai_response(level, emotions):
    suggestions = {
        "Low": "You're doing well! Keep up with your routine and try to encourage others around you.",
        "Medium": "Take a short walk, stretch, or do a 5-minute meditation. Consider journaling your thoughts.",
        "High": "You may benefit from guided breathing exercises, talking to someone, or listening to calming music."
    }
    tip = suggestions.get(level, "Maintain balance and reach out for help when needed.")
    emotion_string = ', '.join(emotions)
    return f"🤖 Based on your emotional state ({emotion_string}), here's a tip: {tip}"

if page == "🧠 AI Support":
    st.subheader("🧠 GPT-Based Emotional Support Bot")

    if st.session_state.prediction_log:
        latest = st.session_state.prediction_log[-1]
        mood_level = latest['Prediction']
        detected_emotions = latest.get('Emotions', 'neutral').split(', ')

        st.markdown(f"### Last Detected Mood Level: **{mood_level}**")
        st.markdown(f"### Detected Emotions: `{', '.join(detected_emotions)}`")

        st.markdown("---")
        st.markdown("#### 🤖 GPT-Based Suggestion:")
        st.success(get_ai_response(mood_level, detected_emotions))

        st.markdown("---")
        st.markdown("#### 📋 Personalized Daily Plan:")
        if mood_level == 'High':
            st.info("🕒 10 mins of deep breathing → Write down 3 things that stress you → Listen to calm music")
        elif mood_level == 'Medium':
            st.info("🕒 15 mins of light exercise → Drink water → Read a page of a book")
        else:
            st.info("🕒 Maintain your good flow → Help a friend → Enjoy your favorite hobby")

    else:
        st.warning("No data found. Please perform a prediction first from the text or survey section.")

def show_avatar(level):
    if level == "Low":
        st.markdown("## 😊 Low Stress")
        st.markdown("You're doing awesome! 🎉 Relax and keep shining!")
        st.image("https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif")

    elif level == "Medium":
        st.markdown("## 😐 Medium Stress")
        st.markdown("Take a deep breath and stretch a little. You've got this!")
        st.image("https://media.giphy.com/media/13borq7Zo2kulO/giphy.gif")

    elif level == "High":
        st.markdown("## 😣 High Stress")
        st.markdown("You are not alone. Breathe and take a break.")
        st.image("https://media.giphy.com/media/xT5LMHxhOfscxPfIfm/giphy.gif")


# 📘 Overview
if page == "📘 Overview":
    st.title("📘 Academic Stress Detector")
    st.markdown("""
Welcome to the **Academic Stress Detector App**!  
This platform supports your mental well-being through a variety of interactive tools:

### 🔍 What You Can Do:
- 📝 **[Analyze Text-Based Stress](#📝-text-analysis)**  
  Reflect on your feelings through simple text and get a stress prediction.

- 📋 **[Predict Stress via Survey](#📋-survey-prediction)**  
  Rate academic pressure, anxiety, and more — and get your stress level instantly.

- 🎭 **[Visual Feedback with Emojis](#📝-text-analysis)**  
  Get engaging emotional avatars based on your input mood level.

- 📊 **[View Stress Trends](#📊-visualization)**  
  See stress patterns over time through bar charts and animated trends.

- 📆 **[Weekly Mood Tracker](#📆-weekly-mood-tracker)**  
  Track how you've felt over the past week.

- 📬 **[Google Calendar Reminders](#📬-calendar-reminder)**  
  Add daily check-in reminders for self-reflection.

- 🎵 **[Listen to Guided Meditation](#🎵-guided-meditation)**  
  Play relaxing music or short meditative sessions directly in the app.

- 📓 **[Keep a Mood Journal](#📓-mood-journal)**  
  Write down how you're feeling today and reflect later.

- 🧪 **[Take a Coping Style Quiz](#🧪-coping-quiz)**  
  Learn how you respond to stress and get a personalized tip.

- 💬 **[Share on the Peer Wall](#💬-peer-support-wall)**  
  Anonymously post your thoughts and connect with others.

- 🧠 **[Chat with the AI Assistant](#🧠-chat-assistant)**  
  Tell the bot how you feel — get a suggestion or just vent.

- 💾 **[Download Your History](#💾-export-predictions)**  
  Export all predictions and journal entries as CSV.

📘 _Start your wellness journey now by selecting a section from the left sidebar._
""")

# 📝 Text Analysis
if page == "📝 Text Analysis":
    st.subheader("📝 Text-Based Stress Detection")
    input_mode = st.radio("Choose input type:", ["One-line Input", "Multi-line Input"])

    if input_mode == "One-line Input":
        text_input = st.text_input("💬 Enter how you feel about your studies today:")
    else:
        text_input = st.text_area("💬 Write in detail how you feel today about your studies:")

    if text_input:
        blob = TextBlob(text_input)
        polarity = blob.sentiment.polarity

        st.markdown(f"**Sentiment Score (Polarity):** `{round(polarity, 2)}`")

        if polarity < -0.3:
            level = 'High'
            reason = "Highly negative emotion detected."
        elif polarity < 0.2:
            level = 'Medium'
            reason = "Some concerns or uncertainty present."
        else:
            level = 'Low'
            reason = "Positive or stable emotional tone."

        st.success(f"Predicted Stress Level (Text): **{level}**")
        st.info(f"**Why?** → {reason}")
        show_avatar(level)

        # 🎁 Gift message for engagement
        if level == "High":
            st.markdown("🎁 **Your Support Pack:** A warm hug, a breathing exercise app, and a playlist of soothing music 🎧💙")
        elif level == "Medium":
            st.markdown("🎁 **Your Balance Boost:** A cup of tea ☕, 10 minutes of calm breathing 🌬️, and a motivational quote 💪")
        else:
            st.markdown("🎁 **Your Wellness Reward:** A pat on the back 🏅, a happy song 🎵, and a relaxing walk suggestion 🚶‍♂️")


# 🔎 Keyword Extraction
        words = words = text_input.lower().split()
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in words if word.isalpha() and word not in stop_words]
        most_common = Counter(keywords).most_common(5)
        keyword_list = [word for word, _ in most_common]
        st.markdown(f"**Top Keywords:** `{', '.join(keyword_list)}`")

        # 🎯 Multi-label Emotion Classification
        emotions_map = {
            "joy": ["happy", "joy", "excited", "love"],
            "anger": ["angry", "mad", "furious"],
            "fear": ["afraid", "scared", "fear", "anxious"],
            "sadness": ["sad", "unhappy", "depressed"]
        }
        detected_emotions = [emotion for emotion, words in emotions_map.items() if any(word in text_input.lower() for word in words)]
        if not detected_emotions:
            detected_emotions = ["neutral"]
        st.markdown(f"**Detected Emotions:** `{', '.join(detected_emotions)}`")

        # 🤖 GPT-like Suggestions (simulated)
        if level == 'High':
            st.warning("🤖 Suggestion: Try journaling, deep breathing, or reaching out to a friend.")
        elif level == 'Medium':
            st.info("🤖 Suggestion: Take a walk or engage in light stretching.")
        else:
            st.success("🤖 Suggestion: Stay consistent and keep taking care of yourself!")

        st.session_state.prediction_log.append({
            "Source": "Text",
            "Input": text_input,
            "Prediction": level,
            "Polarity": round(polarity, 2),
            "Emotions": ', '.join(detected_emotions),
            "Keywords": ', '.join(keyword_list),
            "Timestamp": datetime.now()
        })

# 📋 Survey Prediction
if page == "📋 Survey Prediction" and model:
    st.subheader("📋 Survey-Based Stress Prediction")
    with st.form("survey_form"):
        academic_pressure = st.slider("Academic pressure (0–10)", 0, 10, 5)
        academic_workload = st.slider("Academic workload (0–10)", 0, 10, 5)
        anxiety = st.slider("Anxiety level (0–10)", 0, 10, 5)
        depression = st.slider("Depressive feelings (0–10)", 0, 10, 5)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame({
            'academic_pressure': [academic_pressure],
            'academic_workload': [academic_workload],
            'anxiety': [anxiety],
            'depression': [depression]
            })

    # 🔍 Calculate custom stress score and label
        input_df['stress_score'] = input_df.mean(axis=1)
        st.write(f"Calculated stress score: {input_df['stress_score'].values[0]}")  
        input_df['stress_level'] = input_df['stress_score'].apply(label_stress)
        prediction = input_df['stress_level'].values[0]

        st.success(f"Predicted Stress Level (Survey): **{prediction}**")
        show_avatar(prediction)

        if prediction == "High":
            st.markdown("🎁 **Your Support Pack:** A safe space to talk 🗣️, calming music 🎶, and journaling time 📝")
        elif prediction == "Medium":
            st.markdown("🎁 **Your Balance Boost:** A coffee break ☕, nature sounds 🌳, and a motivational video 🎥")
        else:
            st.markdown("🎁 **Your Wellness Reward:** A smile badge 😊, dance session 🕺, and a gratitude journal 📔")


        # 🤖 AI Suggestion based on survey prediction
        if prediction == 'High':
            st.warning("🤖 Suggestion: Schedule regular breaks, reduce workload if possible, and talk to a counselor.")
        elif prediction == 'Medium':
            st.info("🤖 Suggestion: Practice mindfulness or keep a journal to manage mid-level stress.")
        else:
            st.success("🤖 Suggestion: You're on track! Maintain your routine and stay consistent.")

        st.session_state.prediction_log.append({
            "Source": "Survey",
            "Input": f"P:{academic_pressure}, W:{academic_workload}, A:{anxiety}, D:{depression}",
            "Prediction": prediction,
            "Timestamp": datetime.now()
        })

# 📊 Visualization
if page == "📊 Visualization":
    st.subheader("📊 Stress Predictions Dashboard")

    # --- Reset Button ---
    if st.button("🧹 Reset Prediction Log"):
        st.session_state.prediction_log = []
        st.success("Prediction log has been reset.")

    if st.session_state.prediction_log:
        df_pred = pd.DataFrame(st.session_state.prediction_log)
        df_pred['Timestamp'] = pd.to_datetime(df_pred['Timestamp'])
        df_pred['Date'] = df_pred['Timestamp'].dt.date

        # --- Dropdown Filter ---
        source_filter = st.selectbox("🔽 Filter by Source", ["All", "Text", "Survey"])
        if source_filter != "All":
            df_pred = df_pred[df_pred["Source"] == source_filter]

        # --- Bar Chart (Grouped Count) ---
        grouped_df = df_pred.groupby(['Prediction', 'Source']).size().reset_index(name='Count')
        all_levels = ['Low', 'Medium', 'High']
        all_sources = ['Text', 'Survey']
        full_index = pd.MultiIndex.from_product([all_levels, all_sources], names=["Prediction", "Source"])
        grouped_df = grouped_df.set_index(['Prediction', 'Source']).reindex(full_index, fill_value=0).reset_index()

        fig_bar = px.bar(
            grouped_df,
            x='Source',
            y='Count',
            color='Prediction',
            barmode='group',
            color_discrete_map={
                "Low": "#b9fbc0",
                "Medium": "#fcd5ce",
                "High": "#f08080"
            },
            text='Count',
            title="📊 Stress Levels by Source"
        )
        fig_bar.update_layout(
            xaxis_title="Prediction Source",
            yaxis_title="Number of Predictions",
            legend_title="Stress Level",
            bargap=0.25,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Animated Line Chart: Stress Over Time ---
        trend_df = df_pred.copy()
        trend_df['Score'] = trend_df['Prediction'].map({"Low": 1, "Medium": 2, "High": 3})
        fig_trend = px.line(
            trend_df,
            x='Timestamp',
            y='Score',
            color='Source',
            line_shape='spline',
            animation_frame=trend_df['Date'].astype(str),
            markers=True,
            title="📈 Animated Stress Level Trend Over Time"
        )
        fig_trend.update_layout(
            yaxis=dict(
                title="Stress Level Score",
                tickvals=[1, 2, 3],
                ticktext=["Low", "Medium", "High"]
            ),
            xaxis_title="Timestamp",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# 🤖 AI Insight Summary
        st.subheader("🤖 AI Insight Based on Trends")

        high_logs = [entry for entry in st.session_state.prediction_log[-5:] if entry["Prediction"] == "High"]
        if len(high_logs) >= 3:
            st.error("🚨 Multiple High Stress entries detected!")
            st.markdown("If you're feeling overwhelmed, please don't hesitate to reach out.")
            st.markdown("[📬 Contact University Counselor](mailto:counselor.lpu@support.edu)")
            st.markdown("[🌐 WHO Mental Health Resources](https://www.who.int/teams/mental-health-and-substance-use)")
            st.markdown("[🤝 Talk to a Peer](#💬 Peer Support Wall)")

        if not df_pred.empty:
            latest_score = trend_df.iloc[-1]["Score"]
        if latest_score == 3:
            st.warning("😥 Recent trend shows high stress. Suggest seeking support and lightening your schedule.")
        elif latest_score == 2:
            st.info("😐 Moderate stress detected in recent trends. Suggest mindfulness or routine check-ins.")
        else:
            st.success("😊 Great job! Low stress trend detected. Keep up the positive habits.")

        with st.expander("📋 Show Prediction Log"):
            st.dataframe(df_pred)

    else:
        st.info("No predictions to display yet.")

# 📆 Weekly Mood Tracker
if page == "📆 Weekly Mood Tracker":
    st.subheader("📆 Weekly Mood Tracker")
    if st.session_state.prediction_log:
        df = pd.DataFrame(st.session_state.prediction_log)
        df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
        last_7_days = datetime.now().date() - timedelta(days=6)
        recent_df = df[df['Date'] >= last_7_days]

        mood_trend = recent_df.groupby(['Date', 'Prediction']).size().unstack().fillna(0)
        st.line_chart(mood_trend)
        st.markdown("_Track your stress trend for the last 7 days._")

        # --- 📅 Calendar Heatmap ---
        st.subheader("📅 Calendar Heatmap of Stress Levels")
        heatmap_df = pd.DataFrame(st.session_state.prediction_log)
        heatmap_df['Date'] = pd.to_datetime(heatmap_df['Timestamp']).dt.date

        mood_to_score = {"Low": 1, "Medium": 2, "High": 3}
        heatmap_df["Score"] = heatmap_df["Prediction"].map(mood_to_score)

        calendar_data = heatmap_df.groupby("Date")["Score"].mean().reset_index()
        calendar_data["Date"] = pd.to_datetime(calendar_data["Date"])

        fig = px.density_heatmap(
            calendar_data,
            x="Date",
            y=["Stress Level"] * len(calendar_data),
            z="Score",
            nbinsx=30,
            color_continuous_scale=["#b9fbc0", "#fcd5ce", "#f08080"],
            title="📅 Calendar Heatmap of Stress Levels",
        )
        st.plotly_chart(fig)
    else:
        st.warning("Not enough data. Make predictions first.")

# 📬 Google Calendar Reminder
if page == "📬 Calendar Reminder":
    st.subheader("📅 Add a Reminder on Google Calendar")
    today = datetime.now().strftime("%Y%m%dT090000Z")
    end = (datetime.now() + timedelta(hours=1)).strftime("%Y%m%dT100000Z")
    link = f"https://calendar.google.com/calendar/render?action=TEMPLATE&text=Check+Your+Stress+Level&dates={today}/{end}&details=Self-care+is+important!+Reflect+on+your+day.&location=Online&sf=true"

    st.markdown("Click below to schedule a daily reminder to reflect on your stress levels:")
    st.markdown(f"[📅 Add to Google Calendar]({link})")

# 🧠 Chat Assistant
if page == "🧠 Chat Assistant":
    st.subheader("🧠 GPT-Based Chat Assistant")
    st.markdown("Tell me what's on your mind today 💬")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_msg = st.text_input("👤 You:", key="chat_input")
    if st.button("Send"):
        if user_msg:
            # You can expand this using OpenAI API
            if "tired" in user_msg.lower():
                response = "Try a quick walk or short nap."
            elif "anxious" in user_msg.lower():
                response = "Take 5 deep breaths. You're safe."
            elif "happy" in user_msg.lower():
                response = "Keep doing what works! 😊"
            else:
                response = "Thanks for sharing. You're not alone."

            st.session_state.chat_history.append(("You", user_msg))
            st.session_state.chat_history.append(("AI", response))

    for sender, msg in st.session_state.chat_history[-10:]:
        if sender == "You":
            st.markdown(f"**👤 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")

# 💬 Peer Support Wall
if page == "💬 Peer Support Wall":
    st.subheader("💬 Anonymous Peer Support Wall")
    st.markdown("Share how you're feeling anonymously.")

    if "peer_wall" not in st.session_state:
        st.session_state.peer_wall = []

    with st.form("peer_form"):
        shared_thought = st.text_area("How do you feel today?")
        submitted = st.form_submit_button("Share Anonymously")
        if submitted and shared_thought.strip():
            st.session_state.peer_wall.append({"message": shared_thought.strip(), "👍": 0, "💬": 0})

    for i, post in enumerate(reversed(st.session_state.peer_wall)):
        st.write(f"**Anonymous {len(st.session_state.peer_wall) - i}:** {post['message']}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"👍 {post['👍']}", key=f"like_{i}"):
                post['👍'] += 1
        with col2:
            if st.button(f"💬 {post['💬']}", key=f"comment_{i}"):
                post['💬'] += 1

# 🧩 Mind Games
if page == "🧩 Mind Games":
    st.title("🧩 Mind Games for Mental Wellbeing")
    st.markdown("Play mini-games to **relax, reflect, and reset your mind!** 🎮")

    # 1️⃣ Mood Matching Game
    st.subheader("🎭 Mood Matching Game")
    emotion_words = {"Happy": "😊", "Sad": "😢", "Angry": "😡", "Calm": "😌"}
    question = st.selectbox("Which emoji matches this word: **Happy**", ["😢", "😊", "😡", "😌"])
    if question:
        if question == emotion_words["Happy"]:
            st.success("Correct! Great emotional awareness.")
        else:
            st.error("Oops! Try again next time.")

    st.markdown("---")

    # 2️⃣ Breathing Exercise
    st.subheader("🧘 Breathe with Me")
    if st.button("Start 1-Minute Breathing"):
        breathe = st.empty()
        import time
        for _ in range(3):
            breathe.markdown("### 🌬️ Inhale slowly...")
            time.sleep(4)
            breathe.markdown("### 😮‍💨 Exhale gently...")
            time.sleep(4)
        st.success("Great job! You're calmer now.")

    st.markdown("---")

    # 3️⃣ Affirmation Spinner
    st.subheader("🎡 Spin the Affirmation Wheel")
    affirmations = [
        "🌟 You are enough just as you are.",
        "💪 You’ve overcome 100% of your bad days.",
        "🌈 Every emotion is valid.",
        "🌻 You are doing your best, and that’s okay.",
        "📖 Your story isn’t finished yet."
    ]
    import random
    if st.button("🎯 Spin Now"):
        st.info(random.choice(affirmations))

    st.markdown("---")

    # 4️⃣ Gratitude Journal
    st.subheader("📒 Gratitude Journal")
    with st.form("gratitude_game"):
        st.write("List 3 things you're grateful for today:")
        g1 = st.text_input("🌞 First:")
        g2 = st.text_input("👨‍👩‍👧 Second:")
        g3 = st.text_input("🌻 Third:")
        gratitude_submit = st.form_submit_button("Submit")
        if gratitude_submit:
            if g1 or g2 or g3:
                st.success("That's beautiful! Practicing gratitude boosts happiness. 💖")
                st.balloons()
            else:
                st.warning("Try entering at least one item.")

    st.markdown("---")

    # 5️⃣ Color Your Mood
    st.subheader("🎨 Color Your Mood")
    picked_color = st.color_picker("Pick a color that matches your current emotion:")
    if picked_color:
        st.markdown(f"You chose: `{picked_color}`")
        color_meanings = {
            "#FF0000": "Anger or Passion",
            "#00FF00": "Peace or Hope",
            "#0000FF": "Sadness or Serenity",
            "#FFFF00": "Joy or Energy"
        }
        color_meaning = color_meanings.get(picked_color.upper(), "a unique emotion!")
        st.info(f"This color may represent **{color_meaning}**.")

# 🎵 Sound Therapy
if page == "🎵 Sound Therapy":
    st.title("🎵 Sound Therapy & Mind Relaxation")
    st.markdown("""
    Welcome to the **Sound Therapy Room** 🎧  
    Choose a relaxing sound, sit back, and give yourself a moment of peace. 🌿  
    These audio tracks help reduce anxiety, improve focus, and promote emotional balance.
    """)

    sound_option = st.selectbox("Pick a relaxing vibe 🎼", [
        "🌊 Ocean Waves",
        "🌧️ Rain on Leaves",
        "🔥 Fireplace Crackle",
        "🎶 Soft Piano",
        "🧘 Guided Meditation"
    ])

    sound_links = {
        "🌊 Ocean Waves": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "🌧️ Rain on Leaves": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "🔥 Fireplace Crackle": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "🎶 Soft Piano": "https://www.bensound.com/bensound-music/bensound-slowmotion.mp3",
        "🧘 Guided Meditation": "https://www.bensound.com/bensound-music/bensound-sweet.mp3"
    }

    st.markdown(f"""
    <audio controls autoplay loop>
        <source src="{sound_links[sound_option]}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """, unsafe_allow_html=True)

    st.image("https://media.giphy.com/media/dzaUX7CAG0Ihi/giphy.gif", width=300)
    st.markdown("🕊️ *Take a deep breath...*")

    # 🌬️ Breathing Bubble Animation
    st.markdown("## 🧘 Breathing Bubble: Follow the Rhythm")
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <div class="breathing-circle"></div>
    </div>

    <style>
    @keyframes breathe {
      0% { transform: scale(1); }
      25% { transform: scale(1.3); }
      50% { transform: scale(1.6); }
      75% { transform: scale(1.3); }
      100% { transform: scale(1); }
    }
    .breathing-circle {
      width: 120px;
      height: 120px;
      background: radial-gradient(circle, #a0e7e5, #b4f8c8);
      border-radius: 50%;
      animation: breathe 8s infinite;
      box-shadow: 0 0 25px rgba(0,0,0,0.1);
    }
    </style>

    <p style='text-align:center; font-size:18px;'>Inhale... and Exhale slowly 🌬️</p>
    """, unsafe_allow_html=True)

    # ⏳ Mindfulness Timer
    st.markdown("### ⏳ 1-Minute Mindfulness Timer")
    import time
    if st.button("🧘‍♀️ Start Timer"):
        st.info("Timer started... Inhale deeply. 🌬️")
        with st.spinner("Stay mindful..."):
            for remaining in range(60, 0, -1):
                st.markdown(f"⏳ `{remaining} seconds left...`")
                time.sleep(1)
            st.success("✨ 1 Minute Complete! How do you feel?")

    # ✨ Benefits
    st.markdown("---")
    st.markdown("""
    ### ✨ Benefits of Sound Therapy:
    - Reduces stress and anxiety  
    - Enhances focus and emotional regulation  
    - Aids better sleep  
    - Encourages mindfulness and calm
    """)

    st.markdown("_Use this space daily to recharge your mind._ 💆‍♀️💆‍♂️")

if page == "📓 Mood Journal":
    st.subheader("📓 Daily Mood Journal")
    
    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    today_date = datetime.now().strftime("%Y-%m-%d")
    entry = st.text_area("How are you feeling today?", height=150)

    if st.button("Save Entry"):
        if entry.strip():
            st.session_state.journal_entries.append({
                "Date": today_date,
                "Entry": entry.strip()
            })
            st.success("Journal entry saved!")

    if st.session_state.journal_entries:
        st.markdown("### 📜 Your Past Journal Entries")
        journal_df = pd.DataFrame(st.session_state.journal_entries)
        st.dataframe(journal_df)

        # Export option
        csv = journal_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Journal CSV", csv, file_name="mood_journal.csv", mime="text/csv")

if page == "🧪 Coping Quiz":
    st.subheader("🧪 What’s Your Coping Style?")
    st.markdown("Answer a few fun questions to understand your stress coping style!")

    q1 = st.radio("When you’re stressed, you usually:", [
        "Take a walk or get fresh air",
        "Talk to a friend or write",
        "Eat something or watch a show",
        "Do nothing and bottle it up"
    ])
    q2 = st.radio("Which activity relaxes you the most?", [
        "Exercise or dance",
        "Writing or drawing",
        "Listening to music",
        "Sleeping or laying down"
    ])
    q3 = st.radio("How often do you express your emotions?", [
        "Often", "Sometimes", "Rarely", "Never"
    ])

    if st.button("Get My Coping Style"):
        if "bottle" in q1 or q3 == "Never":
            style = "Avoidant"
            tip = "Try to express feelings through journaling or talk therapy."
        elif "walk" in q1 or q2 == "Exercise or dance":
            style = "Physical"
            tip = "Stay active! It’s a great way to manage stress."
        elif "Talk" in q1 or q2 == "Writing or drawing":
            style = "Emotional"
            tip = "Creative outlets are your strength. Keep expressing!"
        else:
            style = "Passive"
            tip = "Try short mindful breaks and goal-setting to build routine."

        st.success(f"**Your Coping Style:** {style}")
        st.info(f"🧠 Tip: {tip}")

# 💾 Export
if page == "💾 Export Predictions":
    st.subheader("💾 Download Prediction History")
    if st.session_state.prediction_log:
        df = pd.DataFrame(st.session_state.prediction_log)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="stress_predictions.csv", mime="text/csv")
    else:
        st.warning("No predictions yet.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("📬 Made with ❤️ to support student wellbeing.")
