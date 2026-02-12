import os
import io
import json
import uuid
import shutil
import zipfile
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, send_file
from app.config import is_valid_audio_id as is_valid_uuid

bp = Blueprint('profiles', __name__, url_prefix='/api/profiles')


def get_profiles_dir():
    """Get the profiles directory path."""
    profiles_dir = os.path.join(os.path.dirname(current_app.config['UPLOAD_FOLDER']), 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    return profiles_dir


@bp.route('', methods=['GET'])
def list_profiles():
    """List all saved voice profiles."""
    profiles_dir = get_profiles_dir()
    profiles = []

    for filename in os.listdir(profiles_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(profiles_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    profiles.append({
                        'id': data.get('id'),
                        'name': data.get('name'),
                        'sample_count': len(data.get('samples', [])),
                        'created_at': data.get('created_at'),
                    })
            except (json.JSONDecodeError, IOError):
                continue

    # Sort by name
    profiles.sort(key=lambda p: p.get('name', '').lower())
    return jsonify(profiles)


@bp.route('/<profile_id>', methods=['GET'])
def get_profile(profile_id):
    """Get a specific voice profile."""
    if not is_valid_uuid(profile_id):
        return jsonify({'error': 'Invalid profile ID'}), 400
    profiles_dir = get_profiles_dir()
    filepath = os.path.join(profiles_dir, f"{profile_id}.json")

    if not os.path.exists(filepath):
        return jsonify({'error': 'Profile not found'}), 404

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, IOError) as e:
        return jsonify({'error': str(e)}), 500


@bp.route('', methods=['POST'])
def create_profile():
    """Create a new voice profile from current samples."""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    name = data.get('name', '').strip()
    samples = data.get('samples', [])  # [{id, transcript}, ...]

    if not name:
        return jsonify({'error': 'Profile name is required'}), 400
    if not isinstance(samples, list) or len(samples) > 100:
        return jsonify({'error': 'samples must be a list (max 100)'}), 400
    if not samples:
        return jsonify({'error': 'At least one sample is required'}), 400

    profile_id = str(uuid.uuid4())
    profiles_dir = get_profiles_dir()

    # Create profile directory for audio files
    profile_audio_dir = os.path.join(profiles_dir, profile_id)
    os.makedirs(profile_audio_dir, exist_ok=True)

    # Copy audio files to profile directory
    upload_folder = current_app.config['UPLOAD_FOLDER']
    profile_samples = []

    for sample in samples:
        sample_id = sample.get('id')
        if not sample_id or not is_valid_uuid(sample_id):
            continue
        transcript = sample.get('transcript', '')

        src_path = os.path.join(upload_folder, f"{sample_id}.wav")
        if os.path.exists(src_path):
            # Touch mtime to prevent stale-upload cleanup from deleting mid-copy
            os.utime(src_path, None)
            # Copy to profile directory with new ID
            new_sample_id = str(uuid.uuid4())
            dst_path = os.path.join(profile_audio_dir, f"{new_sample_id}.wav")
            shutil.copy2(src_path, dst_path)

            profile_samples.append({
                'id': new_sample_id,
                'transcript': transcript,
            })

    if not profile_samples:
        return jsonify({'error': 'No valid samples found'}), 400

    # Save profile metadata
    profile_data = {
        'id': profile_id,
        'name': name,
        'samples': profile_samples,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }

    filepath = os.path.join(profiles_dir, f"{profile_id}.json")
    try:
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
    except Exception:
        # Clean up orphaned audio directory on metadata write failure
        shutil.rmtree(profile_audio_dir, ignore_errors=True)
        raise

    return jsonify({
        'id': profile_id,
        'name': name,
        'sample_count': len(profile_samples),
    })


@bp.route('/<profile_id>', methods=['DELETE'])
def delete_profile(profile_id):
    """Delete a voice profile."""
    if not is_valid_uuid(profile_id):
        return jsonify({'error': 'Invalid profile ID'}), 400
    profiles_dir = get_profiles_dir()
    filepath = os.path.join(profiles_dir, f"{profile_id}.json")
    profile_audio_dir = os.path.join(profiles_dir, profile_id)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Profile not found'}), 404

    try:
        # Delete metadata file
        os.unlink(filepath)

        # Delete audio files directory
        if os.path.exists(profile_audio_dir):
            shutil.rmtree(profile_audio_dir)

        return jsonify({'success': True})
    except IOError as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/<profile_id>/load', methods=['POST'])
def load_profile(profile_id):
    """Load a profile's samples into the upload folder for use."""
    if not is_valid_uuid(profile_id):
        return jsonify({'error': 'Invalid profile ID'}), 400
    profiles_dir = get_profiles_dir()
    filepath = os.path.join(profiles_dir, f"{profile_id}.json")
    profile_audio_dir = os.path.join(profiles_dir, profile_id)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Profile not found'}), 404

    try:
        with open(filepath, 'r') as f:
            profile_data = json.load(f)

        upload_folder = current_app.config['UPLOAD_FOLDER']
        loaded_samples = []

        for sample in profile_data.get('samples', []):
            sample_id = sample.get('id')
            if not sample_id or not is_valid_uuid(sample_id):
                continue
            transcript = sample.get('transcript', '')

            src_path = os.path.join(profile_audio_dir, f"{sample_id}.wav")
            if os.path.exists(src_path):
                # Copy to uploads with new ID for this session
                new_id = str(uuid.uuid4())
                dst_path = os.path.join(upload_folder, f"{new_id}.wav")
                shutil.copy2(src_path, dst_path)

                loaded_samples.append({
                    'id': new_id,
                    'transcript': transcript,
                })

        return jsonify({
            'profile_name': profile_data.get('name'),
            'samples': loaded_samples,
        })

    except (json.JSONDecodeError, IOError) as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/<profile_id>/export', methods=['GET'])
def export_profile(profile_id):
    """Export a profile as a ZIP file."""
    if not is_valid_uuid(profile_id):
        return jsonify({'error': 'Invalid profile ID'}), 400
    profiles_dir = get_profiles_dir()
    filepath = os.path.join(profiles_dir, f"{profile_id}.json")
    profile_audio_dir = os.path.join(profiles_dir, profile_id)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Profile not found'}), 404

    try:
        with open(filepath, 'r') as f:
            profile_data = json.load(f)

        # Create ZIP in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add metadata
            zf.writestr('profile.json', json.dumps(profile_data, indent=2))

            # Add audio files
            for sample in profile_data.get('samples', []):
                audio_path = os.path.join(profile_audio_dir, f"{sample['id']}.wav")
                if os.path.exists(audio_path):
                    zf.write(audio_path, f"audio/{sample['id']}.wav")

        zip_buffer.seek(0)

        # Sanitize filename
        safe_name = "".join(c for c in profile_data['name'] if c.isalnum() or c in (' ', '-', '_')).strip() or "profile"

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{safe_name}.zip"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/import', methods=['POST'])
def import_profile():
    """Import a profile from a ZIP file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        profiles_dir = get_profiles_dir()

        with zipfile.ZipFile(file, 'r') as zf:
            # Validate no path traversal in zip entries
            for member in zf.namelist():
                if os.path.isabs(member) or '..' in member:
                    return jsonify({'error': 'Invalid ZIP file: contains unsafe paths'}), 400

            if 'profile.json' not in zf.namelist():
                return jsonify({'error': 'Invalid profile ZIP: missing profile.json'}), 400

            profile_data = json.loads(zf.read('profile.json'))

            # Validate required fields
            if not isinstance(profile_data.get('name'), str) or not profile_data['name'].strip():
                return jsonify({'error': 'Invalid profile: missing or empty name'}), 400
            if not isinstance(profile_data.get('samples'), list):
                return jsonify({'error': 'Invalid profile: missing samples list'}), 400

            # Generate new ID for imported profile
            new_id = str(uuid.uuid4())
            profile_audio_dir = os.path.join(profiles_dir, new_id)
            os.makedirs(profile_audio_dir, exist_ok=True)

            try:
                # Extract and rename audio files
                new_samples = []
                for sample in profile_data.get('samples', []):
                    sample_id = sample.get('id', '')
                    if not sample_id or not is_valid_uuid(sample_id):
                        continue
                    old_audio_name = f"audio/{sample_id}.wav"
                    if old_audio_name in zf.namelist():
                        new_sample_id = str(uuid.uuid4())
                        audio_data = zf.read(old_audio_name)
                        new_audio_path = os.path.join(profile_audio_dir, f"{new_sample_id}.wav")
                        with open(new_audio_path, 'wb') as f:
                            f.write(audio_data)
                        new_samples.append({
                            'id': new_sample_id,
                            'transcript': str(sample.get('transcript', ''))[:1000],
                        })

                if not new_samples:
                    shutil.rmtree(profile_audio_dir, ignore_errors=True)
                    return jsonify({'error': 'No valid samples found in ZIP'}), 400

                # Save new profile metadata
                new_profile = {
                    'id': new_id,
                    'name': profile_data['name'].strip()[:200] + ' (imported)',
                    'samples': new_samples,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                }

                filepath = os.path.join(profiles_dir, f"{new_id}.json")
                with open(filepath, 'w') as f:
                    json.dump(new_profile, f, indent=2)
            except Exception:
                shutil.rmtree(profile_audio_dir, ignore_errors=True)
                raise

            return jsonify({
                'id': new_id,
                'name': new_profile['name'],
                'sample_count': len(new_samples),
            })

    except zipfile.BadZipFile:
        return jsonify({'error': 'Invalid ZIP file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
