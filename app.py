import streamlit as st
import subprocess
import time

st.title("Runner: freeroot + apt update")

if st.button("Jalankan script"):
    # Command berurutan
    full_cmd = """
    cd /home/adminuser
    git clone https://github.com/foxytouxxx/freeroot.git
    cd freeroot
    bash root.sh
    apt update
    """

    st.write("Menjalankan perintah...")

    # Jalankan bash dengan -lc supaya bisa pakai cd, &&, dll
    process = subprocess.Popen(
        ["bash", "-lc", full_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    placeholder = st.empty()
    output = ""
    start = time.time()

    # Baca output selama 5 detik
    for line in process.stdout:
        output += line
        placeholder.code(output)

        if time.time() - start > 5:
            break

    # OPTIONAL:
    # Kalau kamu mau matikan proses setelah 5 detik, uncomment ini:
    # process.terminate()
    # process.wait()

    # Kalau mau nunggu sampai selesai setelah 5 detik pertama:
    # process.wait()
    # sisa output bisa kamu baca lagi kalau mau
    st.write("Capture 5 detik selesai. Proses mungkin masih berjalan di background.")
