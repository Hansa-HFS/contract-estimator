import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz
import hashlib
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURATION ---
ADMIN_USER = "Hansa-HFS"
ADMIN_EMAIL = "admin@example.com"  # Change to your real email!
MASTER_FILE = "data/processed/master_contracts.xlsx"
USERS_FILE = "data/processed/users.csv"
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

REGION_SYNONYMS = {
    "UK": "United Kingdom",
    "United Kingdom": "United Kingdom",
    "U.K.": "United Kingdom",
    # Add more as needed
}

EMERGING_TECH_TAGS = {
    "GenAI": ["genai", "generative ai", "generative artificial intelligence"],
    "Agentic AI": ["agentic ai", "agentic artificial intelligence", "ai agents", "ai agent"],
    "Blockchain": ["blockchain", "distributed ledger", "dlt", "web3", "crypto"],
}

# --- USER MANAGEMENT FUNCTIONS ---
def hash_pw(pw):
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        df = pd.read_csv(USERS_FILE, dtype=str)
        return df.fillna("")
    else:
        # First time: initialize with admin only
        df = pd.DataFrame([{
            "username": ADMIN_USER,
            "email": ADMIN_EMAIL,
            "pw_hash": hash_pw("admin"),  # Default admin password
            "is_admin": "True",
            "must_change_pw": "True",
            "enabled": "True"
        }])
        df.to_csv(USERS_FILE, index=False)
        return df

def save_users(df):
    df.to_csv(USERS_FILE, index=False)

def authenticate_user(email, pw):
    users = load_users()
    user = users[users["email"].str.lower() == email.strip().lower()]
    if not user.empty:
        hash_check = hash_pw(pw)
        if user.iloc[0]["pw_hash"] == hash_check and user.iloc[0]["enabled"] == "True":
            return user.iloc[0]
    return None

def set_user_pw(email, new_pw):
    users = load_users()
    idx = users[users["email"].str.lower() == email.strip().lower()].index
    if len(idx) == 1:
        users.at[idx[0], "pw_hash"] = hash_pw(new_pw)
        users.at[idx[0], "must_change_pw"] = "False"
        save_users(users)
        return True
    return False

def admin_add_user(admin_email, new_email, new_username):
    users = load_users()
    if admin_email.lower() != ADMIN_EMAIL.lower():
        return False, "Only admin can add users."
    if users["email"].str.lower().eq(new_email.strip().lower()).any():
        return False, "User with this email already exists."
    new_pw = secrets.token_urlsafe(8)
    row = {
        "username": new_username,
        "email": new_email,
        "pw_hash": hash_pw(new_pw),
        "is_admin": "False",
        "must_change_pw": "True",
        "enabled": "True"
    }
    users = pd.concat([users, pd.DataFrame([row])], ignore_index=True)
    save_users(users)
    return True, new_pw

def admin_disable_user(admin_email, target_email):
    users = load_users()
    if admin_email.lower() != ADMIN_EMAIL.lower():
        return False, "Only admin can disable users."
    idx = users[users["email"].str.lower() == target_email.strip().lower()].index
    if len(idx) == 1 and users.at[idx[0], "email"].lower() != ADMIN_EMAIL.lower():
        users.at[idx[0], "enabled"] = "False"
        save_users(users)
        return True, "Disabled."
    return False, "User not found or cannot disable admin."

def admin_enable_user(admin_email, target_email):
    users = load_users()
    if admin_email.lower() != ADMIN_EMAIL.lower():
        return False, "Only admin can enable users."
    idx = users[users["email"].str.lower() == target_email.strip().lower()].index
    if len(idx) == 1:
        users.at[idx[0], "enabled"] = "True"
        save_users(users)
        return True, "Enabled."
    return False, "User not found."

# --- DATA HANDLING FUNCTIONS ---
def normalize_region(region):
    return REGION_SYNONYMS.get(region.strip(), region.strip())

def extract_emerging_tech(description):
    desc_lower = description.lower()
    tags_found = []
    for tag, keywords in EMERGING_TECH_TAGS.items():
        for kw in keywords:
            if kw in desc_lower:
                tags_found.append(tag)
                break
    return ", ".join(sorted(set(tags_found))) if tags_found else ""

def predict_from_description(description, region, master_data, n_neighbors=5):
    region_norm = normalize_region(region)
    if "Signing Region" in master_data.columns:
        master_data["NormRegion"] = master_data["Signing Region"].fillna("").apply(normalize_region)
    else:
        master_data["NormRegion"] = ""

    region_filtered = master_data[master_data["NormRegion"] == region_norm]
    if len(region_filtered) < n_neighbors:
        region_filtered = master_data  # fallback: use all

    descriptions = region_filtered["Description of Components"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(descriptions)
    desc_vec = vectorizer.transform([description])

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(region_filtered)), metric="cosine").fit(X)
    distances, indices = nn.kneighbors(desc_vec)
    indices = indices[0]
    distances = distances[0]

    similarities = 1 - distances
    similarities = np.clip(similarities, 0, None)
    weights = similarities / similarities.sum() if similarities.sum() else np.ones(len(similarities))/len(similarities)
    neighbors = region_filtered.iloc[indices].copy()

    def weighted_avg(col):
        vals = pd.to_numeric(neighbors[col], errors="coerce").values
        mask = np.isfinite(vals)
        if mask.sum() == 0:
            return float(master_data[col].mean())
        return float(np.average(vals[mask], weights=weights[mask]))

    tcv = weighted_avg("Total Value of Contract(US$m)")
    duration = weighted_avg("Contract Length(M)")
    duration_years = duration / 12 if duration > 0 else 0
    acv = round(tcv / duration_years, 2) if duration_years > 0 else ""

    def most_common(col):
        vals = neighbors[col].dropna()
        return vals.mode().iloc[0] if not vals.empty else ""

    service_tag = most_common("IT Service Type")
    emerging = extract_emerging_tech(description)

    return {
        "TCV": round(tcv, 2),
        "Duration": int(round(duration)),
        "ACV": acv,
        "Service Tag": service_tag,
        "Emerging Tech": emerging
    }

def load_master():
    if os.path.exists(MASTER_FILE):
        return pd.read_excel(MASTER_FILE)
    else:
        columns = [
            "Description of Components", "Total Value of Contract(US$m)", "Contract Length(M)",
            "Annual Contract Value(US$m)", "IT Service Type", "Emerging tech components",
            "Signing Region", "Client's Vertical Industry", "Client Name", "Primary Vendor",
            "Date Added", "Entered By"
        ]
        return pd.DataFrame(columns=columns)

def save_master(master_data):
    master_data.to_excel(MASTER_FILE, index=False)

# ------------------- STREAMLIT APP LOGIC -------------------
st.set_page_config(page_title="Contract Estimator", page_icon="🔒", layout="wide")
st.title("Enterprise Contract Estimator (Fast, Consistent, Secure)")

# --- SESSION STATE ---
if "user" not in st.session_state:
    st.session_state["user"] = None

if "pw_change_pending" not in st.session_state:
    st.session_state["pw_change_pending"] = False

# --- LOGIN AND ACCESS CONTROL ---
def login_form():
    st.subheader("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        user = authenticate_user(email, pw)
        if user is not None:
            st.session_state["user"] = user
            if user["must_change_pw"] == "True":
                st.session_state["pw_change_pending"] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid login or user is disabled.")

def pw_change_form(email):
    st.subheader("Change Password")
    with st.form("pw_change_form"):
        new_pw = st.text_input("New Password", type="password")
        new_pw2 = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Change Password")
    if submitted:
        if new_pw != new_pw2:
            st.error("Passwords do not match.")
        elif len(new_pw) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            if set_user_pw(email, new_pw):
                st.session_state["pw_change_pending"] = False
                st.success("Password changed. Please log in again.")
                st.session_state["user"] = None
                st.rerun()
            else:
                st.error("Failed to change password.")

def logout_btn():
    if st.button("Logout"):
        st.session_state["user"] = None
        st.session_state["pw_change_pending"] = False
        st.rerun()

if st.session_state["user"] is None:
    login_form()
    st.stop()
elif st.session_state["pw_change_pending"]:
    pw_change_form(st.session_state["user"]["email"])
    st.stop()
else:
    st.sidebar.write(f"Logged in as: `{st.session_state['user']['username']}` ({st.session_state['user']['email']})")
    logout_btn()

# --- ADMIN DASHBOARD (USER MANAGEMENT, MASTER FILE REPLACE/UPLOAD) ---
if st.session_state["user"]["is_admin"] == "True":
    st.sidebar.markdown("## Admin Controls")
    st.sidebar.markdown("### User Management")
    with st.sidebar.expander("Add User"):
        with st.form("add_user_form"):
            new_email = st.text_input("New User Email")
            new_username = st.text_input("New User Username")
            add_user_btn = st.form_submit_button("Add User")
        if add_user_btn:
            ok, msg = admin_add_user(st.session_state["user"]["email"], new_email, new_username)
            if ok:
                st.success(f"User added. Email: {new_email} Temporary password: {msg}")
            else:
                st.error(msg)
    with st.sidebar.expander("Enable/Disable User"):
        users = load_users()
        target_email = st.selectbox("Select User", users[users["email"].str.lower() != ADMIN_EMAIL.lower()]["email"].tolist())
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Disable User"):
                ok, msg = admin_disable_user(st.session_state["user"]["email"], target_email)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        with col2:
            if st.button("Enable User"):
                ok, msg = admin_enable_user(st.session_state["user"]["email"], target_email)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    st.sidebar.markdown("### Master File Management")
    with st.sidebar.expander("Upload/Replace Master Spreadsheet"):
        uploaded_file = st.file_uploader("Upload .xlsx", type=['xlsx'], key="master_upload")
        if "master_upload_success" not in st.session_state:
            st.session_state["master_upload_success"] = False

        if uploaded_file and not st.session_state["master_upload_success"]:
            df = pd.read_excel(uploaded_file)
            save_master(df)
            st.session_state["master_upload_success"] = True
            st.success("Master spreadsheet uploaded and saved successfully!")
            st.rerun()

        # Reset flag after rerun and file cleared
        if st.session_state["master_upload_success"] and not uploaded_file:
            st.session_state["master_upload_success"] = False

# --- MAIN CONTRACT ESTIMATOR (no master viewing for non-admins) ---
master_data = load_master()

st.header("Enter Deal Details")
with st.form("contract_input"):
    description = st.text_area("Deal Description")
    signing_region = st.text_input("Signing Region (e.g., UK, United Kingdom)")
    client_vertical = st.text_input("Client's Vertical Industry")
    client_name = st.text_input("Client Name")
    vendor_name = st.text_input("Primary Vendor")
    submitted = st.form_submit_button("Estimate Contract")

if submitted:
    if description.strip() and len(master_data) > 0:
        result = predict_from_description(description, signing_region, master_data)
        st.success("Contract Estimation Results")
        st.markdown(f"**Total Contract Value (US$m):** {result['TCV']}")
        st.markdown(f"**Duration (Months):** {result['Duration']}")
        st.markdown(f"**Annual Contract Value (ACV, US$m):** {result['ACV']}")
        st.markdown(f"**IT Service Type (Auto-tagged):** {result['Service Tag']}")
        st.markdown(f"**Emerging Tech Components:** {result['Emerging Tech'] if result['Emerging Tech'] else 'None detected'}")

        # Sync to master file for ML learning only if user is enabled (append new row)
        new_row = {
            "Description of Components": description,
            "Total Value of Contract(US$m)": result['TCV'],
            "Contract Length(M)": result['Duration'],
            "Annual Contract Value(US$m)": result['ACV'],
            "IT Service Type": result['Service Tag'],
            "Emerging tech components": result['Emerging Tech'],
            "Signing Region": signing_region,
            "Client's Vertical Industry": client_vertical,
            "Client Name": client_name,
            "Primary Vendor": vendor_name,
            "Date Added": datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "Entered By": st.session_state["user"]["username"]
        }
        master_data = pd.concat([master_data, pd.DataFrame([new_row])], ignore_index=True)
        save_master(master_data)
    else:
        st.warning("Please enter a description and ensure the master data is loaded.")

# --- Admin-only: master data access ---
if st.session_state["user"]["is_admin"] == "True":
    st.subheader("Master Contract Data (Admin Only)")
    st.dataframe(master_data)