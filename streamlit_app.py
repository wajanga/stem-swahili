import streamlit as st
import hmac
from anthropic import Anthropic

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
        st.error(f"Hitilafu wakati wa kizazi cha maandishi: {e}")


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

include_visuals = st.checkbox("Pendekeza visaidizi vya kuona")

if st.button("Zalisha Maudhui"):
    with st.spinner("Inazalisha maudhui, tafadhali subiri..."):
        # Generate text
        text = generate_text(topic, difficulty, content_length, include_examples, language_style, include_visuals)
        if text:
            st.subheader("Maandishi Yaliyotengenezwa:")
            st.write(text)
        else:
            st.error("Hitilafu imetokea wakati wa kizazi cha maudhui.")
