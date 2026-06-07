import sys
import os

# Add the project root to sys.path so 'src' can be imported correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api import create_app

if __name__ == "__main__":
    print("Starting BUSI Despeckle & Segment Demo...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app = create_app()
    # threaded=True: a slow /segment request (e.g. heavy despeckle methods on
    # large images) must not block the page itself or other concurrent requests
    # on this single-process dev server.
    app.run(debug=True, port=5000, threaded=True)
