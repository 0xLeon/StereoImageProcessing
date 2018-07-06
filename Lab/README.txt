RealSense D415

Stefan Hahn <stefan.hahn@hu-berlin.de>, Andre Niendorf <niendora@hu-berlin.de>

Ordner:
 - DataAcquisition: Beinhaltet Batchfiles und Symlinks zur Aufnahme von Datensets
 - Plots: Beinhaltet die Plots für die drei Datensets als PNG-Dateien
 - RealSense-D415-Data-*: Die verwendeten Datensets

Dateien:
 - FilterPointCloud.py: Enthält Methoden, um automatisiert Punktwolken vorzuverarbeiten.
   Entfernt Bereiche auf Grundlage von angegeben Filtern und schmeißt Groben Unfug™ direkt raus.

 - GetDepthFrames.py: Zentrales Script, welches mit Hilfe von pyrealsense2 die Daten erzeugt.
   Speichert Punktwolken als PLY, Tiefenbilder als NPZ sowie Kameraintrinsics.
   Das Script bietet mehrere Kommandozeilenparameter, um die Konfiguration der Aufnahmen zu ermöglichen.

 - GetRealsenseProfiles.py: Liest die unterstützten Aufnahmeprofile aller angeschlossenen
   Kameramodule aus. Dies beinhaltet auch Kameras ohne RealSense-Modul.

 - PLYObject.py: Beinhaltet eine Klasse zum Verarbeiten von PLY-Dateien.

 - PointCloudDensity.py: Beinhaltet Funktionen zur Berechnung der Punktdichte einer Punktwolke.
   Dazu zählen insbesonders Methoden zur Berechnung der Punktdichte einer Ebene im Raum.

 - ProcessPlanePointCloud.py: Wendet Filter auf ein Datenset an. Filter können dabei unabhängig
   für verschiedene Entfernungen definiert werden. In die gefilterten Punktwolken werden Ebenen
   gefitted und damit dann der Fehler berechnet. Ebenso werden die Punktdichten berechnet.
   Die Resultate werden in eine pickle-Datei geschrieben, welche im jeweiligen Unterordner des
   Datensets unter dem Namen PointCloudProcessingData.pkl gespeichert wird.

 - pyrealsense2.cp36-win_amd64.pyd: Python-Binding für librealsense2. Erlaubt den Zugriff aus Python
   auf das RealSense-Gerät. Kompiliert für Python 3.6 unter Windows x64.

 - README.txt: Diese Datei.

 - RealSense_D415_PLaneFitting.ipynb: Jupyter Notebook zur Präsenation der Resultate. Dieses greift
   auf die meisten der vorhandenen Python-Skripte zu.

 - realsense2.dll: Native Bibliothek zum Zugriff auf RealSense-Geräte. Dies ist die x64-Version der
   Bibliothek. Wenn diese nicht importiert werden kann, müssen die VC++ 2017 Runtime Libraries
   installiert werden.
