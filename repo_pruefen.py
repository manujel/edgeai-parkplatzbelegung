from pathlib import Path

root = Path(__file__).resolve().parent
required = [
    "01_setup_check.ipynb", "download_data.py", "energievergleich.py", "energievergleich_umgekehrt.py",
    "split_nach_tagen.json", "stichprobe.csv", "testergebnisse.csv",
    "audit_stichprobe.csv", "benchmark_vollbild.csv", "messung_arbeitsspeicher.csv",
    "vergleich_fp32_teilint8.csv", "vergleich_testmodelle.csv",
    "vergleich_validierung.csv", "mobilenetv3_fp32.onnx",
    "mobilenetv3_int8_nur_gemm.onnx",
]
optional = ["mobilenetv3_bestes_modell.pt", "parkplatz_demo_mit_unsicherheit.png",
            "audit_stichprobe.png"]
for name in required:
    path = root / name
    print(("OK     " if path.is_file() and path.stat().st_size else "FEHLT  ") + name)
print("\nOptional:")
for name in optional:
    print(("OK     " if (root / name).is_file() else "FEHLT  ") + name)
