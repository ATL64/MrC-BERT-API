# project/server/__init__.py

import os

from flask import Flask
#from flask_bootstrap import Bootstrap
from flask_cors import CORS
from flask_login import LoginManager

# instantiate the extensions
#bootstrap = Bootstrap()


def create_app(script_info=None):

    # instantiate the app
    app = Flask(__name__)
    CORS(app)

    # set config
    app_settings = os.getenv("APP_SETTINGS")
    app.config.from_object(app_settings)

    # For flask-login
    app.secret_key = b'\xc61\x1b\x90:^d\xee<%Q\x03\xb2\xbf2\x85'
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    # set up extensions
    #bootstrap.init_app(app)

    # register blueprints
    from project.server.main.views import main_blueprint

    app.register_blueprint(main_blueprint)

    # shell context for flask cli
    app.shell_context_processor({"app": app})

    return app
