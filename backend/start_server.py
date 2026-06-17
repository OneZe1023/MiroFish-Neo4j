import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['FLASK_HOST'] = '127.0.0.1'
os.environ['FLASK_PORT'] = '5003'
from app import create_app
app = create_app()
app.run(host='127.0.0.1', port=5003, debug=False, threaded=True)
