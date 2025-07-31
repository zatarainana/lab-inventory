import requests
import socket
from urllib.request import urlopen


def test_connection():
    print("\n=== Network Diagnostic ===\n")

    # Test local server
    try:
        response = requests.get("http://localhost:8086", timeout=2)
        print(f"✅ Local server response: {response.status_code} {response.text}")
    except Exception as e:
        print(f"❌ Local server connection failed: {str(e)}")

    # Test port availability
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", 8086))
        print("✅ Port 8086 is available")
    except OSError as e:
        print(f"❌ Port 8086 is in use: {str(e)}")
    finally:
        sock.close()

    # Test internet access
    try:
        urlopen("https://google.com", timeout=2)
        print("✅ Internet access working")
    except Exception as e:
        print(f"❌ No internet access: {str(e)}")


if __name__ == "__main__":
    test_connection()