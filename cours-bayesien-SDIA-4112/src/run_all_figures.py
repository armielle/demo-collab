"""Execute, dans l'ordre, tous les scripts qui produisent les figures du cours.

Usage : python src/run_all_figures.py [prefixe ...]
Sans argument : tous les scripts src/fig_ch*.py, tries par nom de fichier.
Les threads sont limites (voir figstyle.py) pour ne pas saturer la machine.
"""
import subprocess, sys, os, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable

pat = sys.argv[1:] or ["fig_ch*"]
scripts = []
for p in pat:
    scripts += glob.glob(os.path.join(HERE, p + ".py"))
scripts = sorted(set(s for s in scripts if not s.endswith(("figstyle.py",))))

env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
           MKL_NUM_THREADS="2", PYTENSOR_FLAGS="cxx=", MPLBACKEND="Agg")

t0 = time.time()
ok = 0
for s in scripts:
    t = time.time()
    print(f"\n{'='*70}\n>>> {os.path.basename(s)}\n{'='*70}", flush=True)
    r = subprocess.run([PY, os.path.basename(s)], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=7200)
    tail = (r.stdout or "").strip().splitlines()[-12:]
    for l in tail:
        print("   ", l[:150])
    if r.returncode != 0:
        print("    !!! ECHEC, derniere erreur :")
        for l in (r.stderr or "").strip().splitlines()[-6:]:
            print("      ", l[:150])
    else:
        ok += 1
    print(f"    ({time.time()-t:.1f} s)", flush=True)

print(f"\n{ok}/{len(scripts)} scripts reussis en {time.time()-t0:.0f} s")
