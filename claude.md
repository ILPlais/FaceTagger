# Claude

Un `Traceback` est apparu dans la sortie ; je regarde de quoi il s'agit avant de conclure.

Propre avec une vraie boucle `mainloop`. Je mets à jour le fichier livré.

Presque : tout s'installe avec pip, sauf **ExifTool** (un programme externe) et **Tkinter**, qui n'est pas sur PyPI mais est livré avec Python.

**Dans le venv (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install opencv-python pillow numpy
pip install pillow-heif
```
`pillow-heif` est facultatif (photos HEIC). N'installe pas `opencv-python` et `opencv-python-headless` ensemble.

**Hors pip**
- **Tkinter** : il est inclus dans l'installeur de python.org (case « tcl/tk and IDLE », cochée par défaut) et fonctionne dans un venv. Vérifie avec `python -m tkinter`.
- **ExifTool** : il s'installe avec winget : `winget install --id OliverBetz.ExifTool --exact`. En manuel, télécharge le zip, renomme `exiftool(-k).exe` en `exiftool.exe`, puis place-le avec le dossier `exiftool_files` dans un dossier du PATH. Autre option : `--exiftool "C:\Outils\exiftool\exiftool.exe"`. Contrôle avec `exiftool -ver`.
- **Modèles de détection** : environ 37 Mo, téléchargés automatiquement au premier lancement dans `C:\Users\<toi>\.face_tagger\models`. Si le téléchargement échoue, le message indique l'URL et le dossier où les placer à la main.

**Lancement** : `python face_tagger.py --directory "D:\Photos"`

**Ce qui est testé, et ce qui ne l'est pas**
- Je n'ai pas pu tester sous Windows, seulement sous Linux. L'interface, le regroupement, la lecture et l'écriture des métadonnées (accents compris) et le cycle écriture puis ré-analyse fonctionnent.
- Pour la détection, j'ai utilisé un moteur de substitution, car les vrais modèles ne se téléchargent pas depuis mon environnement. J'ai vérifié que les appels à l'API OpenCV correspondent à la version installée, mais je n'ai pas fait tourner les vrais modèles. Pense à tester d'abord sur un petit dossier, avec la sauvegarde `*_original` cochée.
- Les accents dans les noms de fichiers : la méthode conseillée sous Windows est de passer les noms via un fichier texte UTF-8 avec l'option `-@`, et c'est ce que fait le script.
- Pour Windows, j'ai corrigé deux points dans le fichier mis à jour (`face_tagger.py`) : le JSON temporaire n'est plus passé par un chemin absolu (qui pourrait contenir des accents dans ton nom d'utilisateur), et ExifTool ne fait plus clignoter de fenêtre console.

Si Tk affiche mal les emojis ou plante, relance avec `--no-emoji`. Si ExifTool semble se bloquer, Windows Defender est parfois en cause : ajoute une exclusion pour `exiftool.exe`.
