from waitress import serve

from fvgvisionai.web_controller.app_service import execute_fvgvision_ai
from fvgvisionai.web_controller.web_controller import app

if __name__ == "__main__":
    execute_fvgvision_ai()
    #csrf = CSRFProtect()
    #csrf.init_app(app)
    # Configura la chiave segreta (usa qualcosa di sicuro in produzione)
    #app.config['SECRET_KEY'] = 'your-very-secure-random-key'


serve(app, host='0.0.0.0', port=8081, threads=4)
