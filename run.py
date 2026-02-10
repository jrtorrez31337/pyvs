#!/usr/bin/env python3
"""
Qwen3-TTS Web Application Entry Point

Usage:
    uv run python run.py [--host HOST] [--port PORT] [--no-preload]

Options:
    --host HOST       Host to bind to (default: 0.0.0.0)
    --port PORT       Port to bind to (default: 5000)
    --no-preload      Don't preload models at startup
    --no-ssl          Disable HTTPS (microphone won't work over LAN)
"""
import argparse
import os
import sys
import ssl
from pathlib import Path


def generate_self_signed_cert(cert_dir: Path):
    """Generate a self-signed certificate for HTTPS."""
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        return cert_file, key_file

    print("Generating self-signed SSL certificate...")
    cert_dir.mkdir(parents=True, exist_ok=True)

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        import socket

        # Generate key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Get hostname and IP for SAN
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            local_ip = "127.0.0.1"

        # Build certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])

        import ipaddress
        san_list = [
            x509.DNSName("localhost"),
            x509.DNSName(hostname),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]

        # Add local IP if different from localhost
        if local_ip != "127.0.0.1":
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        # Write files
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"Certificate generated: {cert_file}")
        return cert_file, key_file

    except ImportError:
        print("ERROR: 'cryptography' package required for SSL certificate generation.")
        print("Install it with: uv add cryptography")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Qwen3-TTS Web Application')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--no-preload', action='store_true', help="Don't preload models at startup")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-ssl', action='store_true', help='Disable HTTPS (microphone won\'t work over LAN)')
    args = parser.parse_args()

    # Set environment variables for CUDA
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')
    os.environ.setdefault('HF_HOME', '/data/models/huggingface')

    # Import app after setting environment
    from app import create_app
    from app.services.tts_service import tts_service
    from app.services.stt_service import stt_service
    from app.services.clearvoice_service import clearvoice_service
    from app.services.chatterbox_service import chatterbox_service

    app = create_app()

    if not args.no_preload:
        print("\n" + "="*60)
        print("Loading models... This may take a few minutes.")
        print("="*60 + "\n")

        # Load models
        print("Loading TTS models on GPU 0...")
        tts_service.load_models()

        print("\nLoading STT model on GPU 1...")
        stt_service.load_model()

        print("\nLoading ClearVoice speech enhancement model...")
        clearvoice_service.load_model()

        print("\nLoading Chatterbox multilingual TTS...")
        chatterbox_service.load_model()

        print("\n" + "="*60)
        print("All models loaded! Starting server...")
        print("="*60 + "\n")

    # Setup SSL context for HTTPS (required for microphone access over LAN)
    ssl_context = None
    protocol = "http"

    if not args.no_ssl:
        cert_dir = Path(__file__).parent / "certs"
        cert_file, key_file = generate_self_signed_cert(cert_dir)
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_file, key_file)
        protocol = "https"

        print("\n" + "="*60)
        print("IMPORTANT: Using self-signed certificate for HTTPS")
        print("Your browser will show a security warning - this is expected.")
        print("Click 'Advanced' -> 'Proceed' to accept the certificate.")
        print("="*60 + "\n")

    print(f"Starting server at {protocol}://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        ssl_context=ssl_context
    )


if __name__ == '__main__':
    main()
