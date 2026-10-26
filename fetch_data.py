"""Acquire the public languageR lexdec data with a fixed content checksum."""
from pathlib import Path
import hashlib
import urllib.request

URL = "https://raw.githubusercontent.com/cran/languageR/master/data/lexdec.rda"
SHA256 = "a10242dd3632aa7f287fa173685ba856eebde4411e18f97d345cae05b3526050"
TARGET = Path(__file__).resolve().parent / "data" / "lexdec.rda"

def fetch():
    data = TARGET.read_bytes() if TARGET.exists() else urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("lexdec data checksum differs from the analyzed release")
    TARGET.parent.mkdir(exist_ok=True)
    if not TARGET.exists():
        TARGET.write_bytes(data)
    return TARGET

if __name__ == "__main__":
    print(fetch())
