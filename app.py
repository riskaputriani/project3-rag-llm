import streamlit as st
import subprocess
import shlex

st.title("Subprocess Runner")

# Input command
user_cmd = st.text_input("Masukkan command subprocess", "ping -c 4 google.com")

# Tombol untuk eksekusi
if st.button("Jalankan Command"):
    if not user_cmd.strip():
        st.error("Command tidak boleh kosong.")
    else:
        st.write(f"Menjalankan: `{user_cmd}`")

        # Parse command -> lebih aman daripada shell=True
        try:
            cmd = shlex.split(user_cmd)
        except Exception as e:
            st.error(f"Command error: {e}")
            st.stop()

        # Mulai subprocess
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            output_box = st.empty()
            logs = ""

            # Stream real-time
            for line in process.stdout:
                logs += line
                output_box.code(logs)

            process.wait()

            # Ambil stderr jika ada
            err = process.stderr.read()
            if err:
                st.error(err)

            st.success("Selesai menjalankan command.")

        except FileNotFoundError:
            st.error("Command tidak ditemukan. Pastikan perintah tersedia.")
        except Exception as e:
            st.error(f"Error saat menjalankan subprocess: {e}")
