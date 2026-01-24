from flask import Blueprint, jsonify
from app.services.gpu_service import gpu_service

bp = Blueprint('system', __name__, url_prefix='/api/system')


@bp.route('/gpu', methods=['GET'])
def get_gpu_status():
    """Get GPU status for all devices."""
    gpus = gpu_service.get_gpu_status()
    return jsonify(gpus)
