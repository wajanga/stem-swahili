import streamlit as st
import hmac
from anthropic import Anthropic
from openai import OpenAI
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import scipy.io.wavfile
import io

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Set your Anthropic API key
API_KEY = st.secrets["anthropic_api_key"]
client = Anthropic(api_key=API_KEY)

OPENAI_API_KEY = st.secrets["open_ai_key"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def load_tts_model():
    model = VitsModel.from_pretrained("facebook/mms-tts-swh")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-swh")
    return model, tokenizer

tts_model, tts_tokenizer = load_tts_model()

def generate_text(topic, difficulty, content_length, include_examples, language_style, include_visuals):
    # Construct the prompt based on user inputs
    prompt = f"Kwa Kiswahili {language_style.lower()}, eleza mada ifuatayo kwa mwanafunzi {difficulty}: {topic}."
    
    if include_examples:
        prompt += " Tafadhali jumuisha mifano na shughuli za vitendo."
    
    if include_visuals:
        prompt += " Pendekeza visaidizi vya kuona ambavyo vinaweza kusaidia kuelewa mada hii."
    
    prompt += f" Urefu wa maudhui unapaswa kuwa takriban maneno {content_length}."
    
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=content_length,
            messages=messages
        )
        generated_text = response.content[0].text
        return generated_text.strip()
    except Exception as e:
        st.error(f"Hitilafu wakati wa kuzalisha maandishi: {e}")

def generate_summary(text):
    # Construct a prompt to summarize the text
    summary_prompt = f"Fupisha maelezo yafuatayo kwa Kiswahili: {text}"
    
    # Use the language model to generate the summary
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=150,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary_text = response.content[0].text.strip()
        return summary_text
    except Exception as e:
        st.error(f"Hitilafu wakati wa kuzalisha muhtasari: {e}")
        return None

def text_to_speech(text):
    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform

    # Convert waveform to audio data
    output_numpy = output.cpu().numpy().reshape(-1)
    output_scaled = np.int16(output_numpy * 32767)

    # Save to in-memory buffer
    sample_rate = tts_model.config.sampling_rate
    audio_buffer = io.BytesIO()
    scipy.io.wavfile.write(audio_buffer, rate=sample_rate, data=output_scaled)
    audio_buffer.seek(0)  # Reset buffer pointer to the beginning
    return audio_buffer

def generate_image(prompt):
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"Hitilafu wakati wa kuzalisha picha: {e}")
        return None

st.title("STEM Education Content Generator for Kiswahili")

st.write(
    """
    **Maelekezo**: Ingiza mada ya STEM unayotaka kujifunza. 
    Programu itazalisha maelezo kwa Kiswahili na kutoa sauti ya kusikiliza.
"""
)

topics = [
    "Mzunguko wa maji",              # The water cycle
    "Nishati na aina zake",          # Energy and its forms
    "Umeme na jinsi unavyofanya kazi", # Electricity and how it works
    "Mfumo wa jua",                  # The solar system
    "Mito na maziwa",                # Rivers and lakes
    "Lishe na chakula bora",         # Nutrition and healthy eating
    "Mabadiliko ya hali ya hewa",    # Climate change
    "Mimea na wanyama",              # Plants and animals
    "Afya na usafi",                 # Health and hygiene
    "Uchafuzi wa mazingira",         # Environmental pollution
]

topic = st.selectbox("Chagua mada ya STEM hapa:", topics)

difficulty_levels = ["Darasa la 1-2", "Darasa la 3-4", "Darasa la 5-6", "Darasa la 7"]
difficulty = st.selectbox("Chagua kiwango cha elimu:", difficulty_levels)

content_length = st.slider("Chagua urefu wa maudhui (maneno):", min_value=100, max_value=1000, value=500, step=50)

include_examples = st.checkbox("Jumuisha mifano na shughuli za vitendo")

language_styles = ["Rasmi", "Maongezi"]
language_style = st.selectbox("Chagua mtindo wa lugha:", language_styles)

include_visuals = st.checkbox("Pendekeza picha au mchoro")

if st.button("Zalisha Maudhui"):
    with st.spinner("Inazalisha maudhui, tafadhali subiri..."):
        # Generate text
        text = generate_text(
            topic,
            difficulty,
            content_length,
            include_examples,
            language_style,
            include_visuals
        )
        if text:
            st.subheader("Maandishi Yaliyotengenezwa:")
            st.write(text)
            
            summary_text = generate_summary(text)
            with st.spinner("Inazalisha sauti, tafadhali subiri..."):
                audio_data = text_to_speech(summary_text)
                if audio_data:
                    st.subheader("Sauti:")
                    st.audio(audio_data, format='audio/wav')
                else:
                    st.error("Hitilafu imetokea wakati wa kuzalisha sauti.")

            # Optionally, handle image generation if needed
            if include_visuals:
                with st.spinner("Inazalisha picha, tafadhali subiri..."):
                    image_prompt = f"Picha inayoonyesha {topic} kwa wanafunzi wa shule ya msingi."
                    image_url = generate_image(image_prompt)
                    if image_url:
                        st.subheader("Picha Inayohusiana:")
                        st.image(image_url)
                    else:
                        st.error("Hitilafu imetokea wakati wa kuzalisha picha.")
        else:
            st.error("Hitilafu imetokea wakati wa kuzalisha maudhui.")
