import streamlit as st
import subprocess
import time

st.title("Runner: freeroot + apt update")

if st.button("Jalankan script"):
    # Command berurutan
    full_cmd = """
    set -e
    git clone https://github.com/foxytouxxx/freeroot.git || echo "Repo sudah ada, lanjut..."
    cd freeroot
    bash root.sh
    apt install nano
    echo "done update"
    """

    st.write("Menjalankan perintah...")

    # Jalankan bash dengan -lc supaya bisa pakai cd, &&, dan newline
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

    # Baca output selama maks 5 detik
    for line in process.stdout:
        output += line
        placeholder.code(output)

        if time.time() - start > 5:
            break

    # OPTIONAL: kalau mau hentikan proses setelah 5 detik, bisa gini:
    # process.terminate()
    # process.wait()

    # Kalau mau biarkan proses lanjut di background:
    # jangan terminate, biar saja jalan terus.
    st.write("Capture output 5 detik selesai. Proses mungkin masih berjalan di background.")
