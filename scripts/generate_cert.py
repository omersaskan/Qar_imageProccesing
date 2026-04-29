import datetime
import ipaddress
from cryptography import x509
# ... (rest of imports)
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from pathlib import Path

def generate_self_signed_cert(cert_path, key_path):
    # 1. Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # 2. Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"TR"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Istanbul"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Istanbul"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"MeshySiz"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"192.168.1.205"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        # Our certificate will be valid for 1 year
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(u"192.168.1.205")), x509.DNSName(u"localhost")]),
        critical=False,
    ).sign(key, hashes.SHA256())

    # 3. Write files
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Certificate and Key generated: {cert_path}, {key_path}")

if __name__ == "__main__":
    base = Path(__file__).parent.parent
    certs_dir = base / "certs"
    certs_dir.mkdir(exist_ok=True)
    generate_self_signed_cert(certs_dir / "cert.pem", certs_dir / "key.pem")
