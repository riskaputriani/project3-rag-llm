import streamlit as st
import subprocess
import time
import os

st.title("Runner: freeroot + apt update")

if st.button("Jalankan script"):

    st.write("Menjalankan perintah satu per satu...")

    logs = ""
    placeholder = st.empty()

    def run_cmd(cmd, cwd=None):
        """Jalankan command dengan output real-time, capture 5 detik."""
        nonlocal logs
        st.write(f"▶ Menjalankan: `{cmd}`")

        process = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )

        start = time.time()
        for line in process.stdout:
            logs += line
            placeholder.code(logs)

            # Stop capture setelah 5 detik
            if time.time() - start > 5:
                break

    # 1. Clone repo
    run_cmd("git clone https://github.com/foxytouxxx/freeroot.git")

    # Pastikan folder sudah ada
    freeroot_path = os.path.join(os.getcwd(), "freeroot")

    # 2. Jalankan root.sh
    run_cmd("bash root.sh", cwd=freeroot_path)

    # 3. apt update
    run_cmd("apt update")

    st.success("Capture 5 detik selesai untuk tiap command (proses mungkin masih berjalan).")
