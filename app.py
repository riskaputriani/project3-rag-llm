import subprocess
import streamlit as st

if "proc" not in st.session_state:
    st.session_state.proc = None

if st.button("Start"):
    st.session_state.proc = subprocess.Popen(
        ["sleep", "10"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    st.write("Started!")

if st.button("Check Status"):
    proc = st.session_state.proc
    if proc and proc.poll() is None:
        st.write("Still running...")
    else:
        st.write("Not running / Finished")
