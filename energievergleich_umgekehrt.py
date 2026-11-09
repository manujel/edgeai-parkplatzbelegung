
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

if sys.platform != "darwin":
    raise RuntimeError("Dieses Messskript ist für macOS gedacht.")

projektordner = Path(__file__).resolve().parent
eingabe = np.full((100, 3, 160, 160), 0.5, dtype=np.float32)

optionen = ort.SessionOptions()
optionen.intra_op_num_threads = 1

modelle = {
    "fp32": ort.InferenceSession(
        str(projektordner / "mobilenetv3_fp32.onnx"),
        sess_options=optionen,
        providers=["CPUExecutionProvider"],
    ),
    "teil_int8": ort.InferenceSession(
        str(projektordner / "mobilenetv3_int8_nur_gemm.onnx"),
        sess_options=optionen,
        providers=["CPUExecutionProvider"],
    ),
}

for sitzung in modelle.values():
    for _ in range(4):
        sitzung.run(None, {"bild": eingabe})

time.sleep(5)

def messen(name, sitzung=None):
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as protokoll:
        prozess = subprocess.Popen(
            [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power",
                "-i", "1000", "-n", "20",
            ],
            stdout=protokoll,
            stderr=subprocess.PIPE,
            text=True,
        )

        start = time.perf_counter()
        ende = start + 20.5
        stapel = 0

        if sitzung is None:
            time.sleep(max(0, ende - time.perf_counter()))
        else:
            while time.perf_counter() < ende:
                sitzung.run(None, {"bild": eingabe})
                stapel += 1

        dauer = time.perf_counter() - start
        _, fehler = prozess.communicate(timeout=15)
        protokoll.seek(0)
        rohtext = protokoll.read()

    if prozess.returncode != 0:
        raise RuntimeError(f"powermetrics bei {name}: {fehler}")

    (projektordner / f"energie_{name}.txt").write_text(
        rohtext, encoding="utf-8"
    )

    werte = [
        int(wert) for wert in re.findall(
            r"Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW",
            rohtext,
        )
    ]
    if len(werte) < 15:
        raise RuntimeError(
            f"Bei {name} wurden nur {len(werte)} Leistungswerte gefunden."
        )

    # Übergänge am Anfang und Ende nicht in den Mittelwert aufnehmen
    stabile_werte = werte[1:-1]
    mittel = statistics.mean(stabile_werte)

    print(
        f"{name}: {len(stabile_werte)} Werte, "
        f"Mittel {mittel:.0f} mW, "
        f"{stapel} Stapel à 100 Ausschnitte, "
        f"Dauer {dauer:.1f} s",
        flush=True,
    )

messen("leerlauf_vor")
messen("teil_int8", modelle["teil_int8"])
messen("fp32", modelle["fp32"])
messen("leerlauf_nach")
