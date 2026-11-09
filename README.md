# Edge AI – Parkplatzbelegung mit PKLot

Implementierung zum IU-Projektbericht DLBAIPEAI01_D (Aufgabe 1). Das Notebook dokumentiert die Datenauswahl, das Training eines MobileNetV3-Small-Klassifikators, den ONNX-Export, Quantisierungsversuche und die Evaluation auf einem Apple-Silicon-Mac. Die gespeicherten Notebook-Ausgaben stammen aus dem ursprünglichen Projektlauf; der Code in dieser Kopie nutzt relative Projektpfade und einen versionsgebundenen Datensatzabruf. Das ursprünglich verwendete Notebook bleibt beim Verfasser erhalten.

## Einstieg

1. Python 3.11 und Jupyter verwenden. Die ursprüngliche Umgebung hieß `edgeai-parking`; für einen neuen Python-3.11-Kernel: `python -m pip install -r requirements.txt`. Die historischen Messungen erfolgten unter anderem mit PyTorch 2.14.0 und ONNX Runtime 1.30.0 auf macOS/arm64. Für exakte Vergleiche sollten die lokal installierten Paketversionen protokolliert werden.
2. `01_setup_check.ipynb` im Repository-Verzeichnis öffnen und die Zellen von oben nach unten ausführen. Die erste Download-Zelle lädt `ammarnassanalhajali/pklot-dataset/versions/1` mit KaggleHub; alternativ kann `python download_data.py` ausgeführt werden. Beim ersten Abruf kann eine Kaggle-Anmeldung erforderlich sein. Die ursprünglichen Bilder werden nicht im Repository dupliziert.
3. Die **finale** `stichprobe.csv` entsteht in der Zelle unter „Endgültige Stichprobe und Ausschnitte“. Die vorherige 48-Pixel-Stichprobe war ein Vorversuch und wird überschrieben. Die Bildausschnitte werden im Benutzer-Cache unter `~/.cache/edgeai-parking/ausschnitte_v1/` erzeugt.
4. Training und Export benötigen gegebenenfalls Zeit und laden vortrainierte Torchvision-Gewichte. Wer nur die abgegebenen Ergebnisse prüfen will, kann die gespeicherten CSV-Dateien und ONNX-Modelle ansehen, ohne erneut zu trainieren.

**Datensatz:** [PKLot Dataset auf Kaggle](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset), Version 1, Roboflow-Export in COCO mit `train`, `valid`, `test`. Die verwendete Quelle nennt PKLot (Almeida et al., 2015) und CC BY 4.0. Die historischen Original-Splits wurden für die Auswertung nach 100 Aufnahmetagen neu gruppiert (70/15/15, Seed 42). Verwendet werden Klassen `space-occupied` = `belegt` und `space-empty` = `frei`.

## Dateien

| Datei | Zweck |
| --- | --- |
| `01_setup_check.ipynb`, `download_data.py` | Vollständiger Projektablauf und separater Datensatzabruf |
| `energievergleich.py`, `energievergleich_umgekehrt.py` | Leistungsmessung mit unterschiedlicher Modellreihenfolge |
| `split_nach_tagen.json`, `stichprobe.csv` | Reproduzierbare Zuordnung der Tage und gewählte COCO-Rahmen |
| `testergebnisse.csv`, `audit_stichprobe.csv` | Einzelvorhersagen und manuell geprüfte 20 Beispiele |
| `vergleich_*.csv`, `benchmark_vollbild.csv`, `messung_arbeitsspeicher.csv` | Ergebnis- und Ressourcenmessungen |
| `mobilenetv3_bestes_modell.pt` | Gespeicherter PyTorch-Checkpoint, falls im Projektordner vorhanden |
| `mobilenetv3_*.onnx` | Exportierte und quantisierte Modellvarianten, soweit vorhanden |
| `parkplatz_demo_mit_unsicherheit.png`, `audit_stichprobe.png` | Visualisierte Demo und Audit, falls vorhanden |
| `energie_*.txt` | Rohprotokolle von `powermetrics`, falls vorhanden |

Die CSV-/JSON-Dateien und Modelle müssen aus dem ursprünglichen Projektordner **zusätzlich** in dieses Repository kopiert werden. Sie sind nicht Bestandteil dieses Vorbereitungspakets. `python repo_pruefen.py` zeigt, ob die zentralen Dateien fehlen. Die Rohbilder (etwa 843 MB komprimiert) und die 17.000 lokal generierten PNG-Ausschnitte bleiben außerhalb von Git; der Download und ihre Erzeugung sind im Notebook dokumentiert. `stichprobe.csv` enthält die genaue Abbildung von Rahmen auf Quelldateien und ist Teil der Datenbereitstellung.

## Ergebnis und Grenzen

- Finale Stichprobe: Training 12.000, Validierung 2.000, Test 3.000 Ausschnitte.
- Test gegen die vorhandenen COCO-Klassen: FP32 und Teil-INT8 (nur `Gemm`) jeweils Accuracy 0,907 und F1-macro 0,9033; 0/3.000 unterschiedliche Vorhersagen.
- FP32-Modell ca. 5,81 MiB, Teil-INT8 ca. 4,13 MiB. Vollständige INT8-Quantisierung ergab in dieser Messung deutlich schlechtere Validierungswerte und wurde nicht als bevorzugte Variante übernommen.
- Einige kleine Ausschnitte und Quelllabels sind problematisch; die manuelle Prüfung von 20 Beispielen ist eine gezielte Stichprobe, keine Korrektur des gesamten Testdatensatzes.
- Die Demo klassifiziert **vorgegebene** COCO-Stellplatzrahmen in einem Kamerabild. Eine automatische Lokalisierung der Stellplätze und ein Test auf einer separaten mobilen Hardware gehören nicht zu dieser Implementierung.

## Energievergleich auf macOS

Die beiden Skripte messen CPU+GPU+ANE-Leistungswerte per `powermetrics` für Leerlauf, FP32 und Teil-INT8. Auf dem Mac zuerst im Terminal `sudo -v` ausführen, danach innerhalb der gültigen sudo-Sitzung `python energievergleich.py` und `python energievergleich_umgekehrt.py` starten. Die Modelle müssen im selben Ordner wie die Skripte liegen. Durch Untergrundlast, Reihenfolge und schwankenden Leerlauf sind die Messungen nur als Näherung zu verstehen; ein klarer Energiegewinn wurde daraus nicht abgeleitet.

## Hinweis zur Nachvollziehbarkeit

Die gespeicherten Notebook-Ausgaben zeigen den ursprünglichen Lauf. Bei einer erneuten vollständigen Ausführung können Training und Laufzeiten je nach Gerät, Bibliotheksversion und Systemlast abweichen. Die beiden dokumentierten festen Pfade zum damaligen Benutzerordner wurden in dieser Repository-Kopie durch `Path.cwd()` beziehungsweise den versionsgebundenen KaggleHub-Abruf ersetzt; die Berechnungslogik ist ansonsten beibehalten.
