import numpy as np, sys, urllib.request, os
print("numpy", np.__version__)
try:
    from scipy import datasets
    import scipy
    print("scipy", scipy.__version__)
    img = datasets.face(gray=True)
    print("scipy.face OK", img.shape, img.dtype, img.min(), img.max())
except Exception as e:
    print("scipy.face ECHEC:", type(e).__name__, str(e)[:140])
urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
]
for url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        d = urllib.request.urlopen(req, timeout=25).read()
        print("net OK", len(d), "octets")
    except Exception as e:
        print("net ECHEC:", type(e).__name__, str(e)[:140])
