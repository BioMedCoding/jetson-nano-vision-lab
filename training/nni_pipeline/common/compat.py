"""Shim di compatibilita' filelock <-> ultralytics.

filelock e' fissato a <3.12 (vedi setup_host.sh) perche' nnictl (nni 3.0) va
in deadlock silenzioso con filelock >=3.12 -- il SimplePreemptiveLock interno
di NNI eredita da filelock.SoftFileLock e il suo comportamento e' cambiato in
modo incompatibile tra le versioni (verificato in sessione: un lock legittimo
diventava permanentemente non acquisibile anche oltre la soglia di
staleness).

ultralytics.data.converter fa pero' `from filelock import AsyncFileLock,
Timeout` a livello di modulo (aggiunto in una versione di filelock
successiva alla 3.12), quindi il solo *import* di ultralytics fallisce con
filelock <3.12, anche se il nostro codice non usa mai le funzioni
async-lock-protette di quel modulo (usate solo per un percorso di cache
opzionale che qui non viene mai esercitato). Basta che il nome esista.

Importare questo modulo per side-effect PRIMA di qualunque `import
ultralytics` (diretto o transitivo via yolov5's utils/general.py)."""
import filelock

if not hasattr(filelock, "AsyncFileLock"):
    filelock.AsyncFileLock = filelock.FileLock
