import os
import services
from flask import Flask
from routes import init_routes
from flask_cors import CORS
import debugpy
import config
import debugger
import generate_data_model

# init debugger if needed
debugger.init()

app = Flask(__name__)
CORS(app)

generate_data_model.main()

# Inicializar las rutas
init_routes(app)

if __name__ == "__main__":
    
    app.run(debug=True, host='0.0.0.0', port=5000)
   
