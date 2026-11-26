"""Re-execute tous les notebooks de TP et reecrit leurs sorties.

Usage : python src/run_all_notebooks.py [nb_tp1 nb_tp5 ...]
Sans argument : tous les scripts src/nb_tp*.py, executes un par un
(jamais en parallele : la machine n'a que 8 coeurs, sans GPU).
"""
import subprocess, sys, os, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable

pat = sys.argv[1:] or ["nb_tp*", "nb_projet*"]
scripts = []
for p in pat:
    scripts += glob.glob(os.path.join(HERE, p + ".py"))
scripts = sorted(set(scripts))

env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
           MKL_NUM_THREADS="2", PYTENSOR_FLAGS="cxx=")
env.pop("MPLBACKEND", None)   # le noyau doit garder son backend inline (figures)

t0 = time.time()
ok = 0
for s in scripts:
    t = time.time()
    print(f"\n{'='*70}\n>>> {os.path.basename(s)}\n{'='*70}", flush=True)
    r = subprocess.run([PY, os.path.basename(s)], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=7200)
    for l in (r.stdout or "").strip().splitlines()[-8:]:
        print("   ", l[:150])
    if r.returncode != 0:
        print("    !!! ECHEC :")
        for l in (r.stderr or "").strip().splitlines()[-8:]:
            print("      ", l[:150])
    else:
        ok += 1
    print(f"    ({time.time()-t:.1f} s)", flush=True)

print(f"\n{ok}/{len(scripts)} notebooks regeneres en {time.time()-t0:.0f} s")

# les scripts nb_*.py reecrivent les notebooks depuis zero : ils effacent la page
# « Consignes et bareme ». On la reinsere ici, en une seule fois.
import tp_consignes
print("\n>>> page de consignes")
tp_consignes.main()
