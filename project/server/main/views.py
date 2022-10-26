# project/server/main/views.py

import redis
from rq import Queue, Connection
from rq.worker import Worker, WorkerStatus
from flask import render_template, Blueprint, jsonify, request, current_app

from project.server.main.tasks import extract_keywords, process_question, log_feedback

main_blueprint = Blueprint("main", __name__,)

# To generate uuid:
"""
import uuid
uuid.uuid4()
"""
users = {
    'xxxx@gmail.com': {
        'uuid': 'xxxxx',
        'role': 'admin'
    },
}

# To generate passwords:
"""
import secrets
import string
alphabet = string.ascii_letters + string.digits
password = ''.join(secrets.choice(alphabet) for i in range(20))
"""
user_password_pairs = {
    'xxxxxx@gmail.com': 'xxxxxxxx',
....
}
def validate_user(user, password):
    if user in user_password_pairs.keys() and user_password_pairs[user]==password:
        return True
    return False

@main_blueprint.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    if validate_user(username, password):
        response_object = {"status": "success"}
        return jsonify(response_object), 200
    else:
        return jsonify({"status": "error"}), 401

@main_blueprint.route("/queue", methods=["GET"])
def get_queue_status():
    with Connection(redis.from_url(current_app.config["REDIS_URL"])):
        q = Queue()
        len_q = len(q)
        worker = Worker(current_app.config['QUEUES'])
        len_q += worker.state == WorkerStatus.BUSY
    response_object = {"length": len(q), "job_ids": q.job_ids}
    return jsonify(response_object)

@main_blueprint.route("/keywords", methods=["POST"])
def get_keywords():
    if not validate_user(request.form["username"], request.form["password"]):
        return jsonify({"status": "error"}), 401
    keywords = extract_keywords(request.form["question"])
    return jsonify(keywords)

@main_blueprint.route("/tasks/<task_id>", methods=["GET"])
def get_status(task_id):
    with Connection(redis.from_url(current_app.config["REDIS_URL"])):
        q = Queue()
        task = q.fetch_job(task_id)
    if task:
        if task.get_status()=='failed':
            return jsonify({"status": "error"}), 500
        response_object = {
            "status": "success",
            "data": {
                "task_id": task.get_id(),
                "task_status": task.get_status(),
                "task_result": task.result,
            },
        }
    else:
        return jsonify({"status": "error"}), 500
    return jsonify(response_object)

@main_blueprint.route("/question", methods=["POST"])
def run_task():
    if not validate_user(request.form["username"], request.form["password"]):
        return jsonify({"status": "error"}), 401
    params = {
        "user_id": users[request.form["username"]]["uuid"],
        "question": request.form["question"],
        "keywords": request.form["keywords"],
        "nOutputs": request.form["nOutputs"],
        "min_year": request.form["min_year"],
        "max_year": request.form["max_year"],
        "pubmed": False
    }
    with Connection(redis.from_url(current_app.config["REDIS_URL"])):
        q = Queue()
        task = q.enqueue(process_question, params)
    response_object = {
        "status": "success",
        "data": {
            "task_id": task.get_id()
        }
    }
    return jsonify(response_object), 202

@main_blueprint.route("/question-pubmed", methods=["POST"])
def run_task_pubmed():
    if not validate_user(request.form["username"], request.form["password"]):
        return jsonify({"status": "error"}), 401
    params = {
        "user_id": users[request.form["username"]]["uuid"],
        "question": request.form["question"],
        "nOutputs": request.form["nOutputs"],
        "min_year": request.form["min_year"],
        "max_year": request.form["max_year"],
        "pubmed": True
    }
    with Connection(redis.from_url(current_app.config["REDIS_URL"])):
        q = Queue()
        task = q.enqueue(process_question, params)
    response_object = {
        "status": "success",
        "data": {
            "task_id": task.get_id()
        }
    }
    return jsonify(response_object), 202

@main_blueprint.route("/feedback", methods=["POST"])
def save_feedback():
    if not validate_user(request.form["username"], request.form["password"]):
        return jsonify({"status": "error"}), 401
    params = {
        "user_id": users[request.form["username"]]["uuid"],
        "task_id": request.form["job_id"],
        "feedback_select": request.form["feedback_select"],
        "feedback_answer": request.form["feedback_answer"],
        "feedback_text": request.form["feedback_text"]
    }
    try:
        log_feedback(params)
    except:
        response_object = {"status": "error"}
        return jsonify(response_object), 500
    response_object = {"status": "success"}
    return jsonify(response_object), 200
