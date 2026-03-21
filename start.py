import subprocess
import sys
import time
import os
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

print("AVOCADO - FRC Team 10600 Pit Assistant")
print("Two Steps Ahead")

# Start Ollama in the background
print("  [..] Starting Ollama...")
try:
    if sys.platform == "win32":
        subprocess.Popen(
            ["ollama", "serve"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    time.sleep(4)
    print("[OK] Ollama started")
except FileNotFoundError:
    print("[!!] Ollama not found — make sure it is installed")
    print("[!!] Download from: https://ollama.com")
    input("Press Enter to exit...")
    sys.exit(1)

ip = get_local_ip()
print()
print("[OK] Server starting...")
print("[OK] Local:   http://localhost:8000")
print(f"[OK] Network: http://{ip}:8000")
print()
print("Press Ctrl+C to stop")
print()

# Start the FastAPI server — always from the script's own directory
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    subprocess.run([sys.executable, os.path.join(script_dir, "main.py")], cwd=script_dir)
except KeyboardInterrupt:
    print()
    print("  Server stopped.")