# FaceTagger

Utilitaire de reconnaissance faciale pour photos

[![Build Multi-OS Executables](https://github.com/ILPlais/FaceTagger/actions/workflows/build.yml/badge.svg)](https://github.com/ILPlais/FaceTagger/actions/workflows/build.yml)

FaceTagger parcourt un dossier de photos (sous-dossiers compris), détecte les visages, les **regroupe par personne**, reprend les noms déjà présents dans les métadonnées des photos, vous propose de **nommer les personnes inconnues**, puis **écrit les noms dans les métadonnées** des photos, où les retrouvent l'Explorateur Windows et les logiciels de photos (digiKam, Lightroom…).

## ✨ Fonctionnalités

- 🔍 **Détection et reconnaissance locales** : deux petits modèles ONNX ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) et [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)) exécutés par OpenCV, sans GPU ni service en ligne : les photos ne quittent jamais l'ordinateur.
- 👥 **Regroupement par personne** selon la similarité des visages, avec un seuil réglable.
- 🏷️ **Reprise des noms existants** : régions de visages (MWG ou Microsoft), `PersonInImage` et, au choix, mots-clés.
- 🧩 **Nommage rapide des inconnus**, des plus fréquents aux plus rares ; renommage, fusion de groupes, déplacement ou mise à l'écart d'un visage.
- 💾 **Écriture des noms** dans les métadonnées, suivie d'une relecture de chaque photo pour vérifier.
- 📦 **Sauvegarde** de la version d'origine de chaque photo modifiée (`*_original`) et **restauration** en un clic.
- ⚡ **Analyse incrémentale** : une base SQLite mémorise visages, vignettes et étiquettes ; une photo inchangée n'est ni ré-analysée ni relue.
- 🌓 Thème clair ou sombre (suit Windows), prise en charge des écrans à haute résolution, infobulles sur chaque commande.

Formats pris en charge : JPEG, PNG, TIFF, WebP et, avec `pillow-heif`, HEIC / HEIF.

## 📥 Installation

### Exécutable prêt à l'emploi

Des exécutables pour **Windows**, **Linux** et **macOS** sont joints à chaque [release](https://github.com/ILPlais/FaceTagger/releases). Il faut seulement installer [ExifTool](#exiftool) à part.

### Depuis les sources

Prérequis : **Python 3.11** ou plus récent, avec Tkinter. Sous Windows, Tkinter est fourni par l'installeur de [python.org](https://www.python.org/downloads/) (case « tcl/tk and IDLE ») ; pour vérifier sa présence :

```powershell
python -m tkinter
```

Puis, dans le dossier du dépôt (PowerShell) :

```powershell
# Création de l'environnement virtuel puis installation des dépendances
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --requirement requirements.txt
```

> ⚠️ N'installez pas `opencv-python` et `opencv-python-headless` dans le même environnement : ils entrent en conflit.

### ExifTool

[ExifTool](https://exiftool.org/), de Phil Harvey, lit et écrit les métadonnées : il est **indispensable**.

```powershell
# Windows
winget install --id OliverBetz.ExifTool --exact
```

```bash
# Linux (Debian, Ubuntu)
sudo apt install libimage-exiftool-perl
```

```bash
# macOS
brew install exiftool
```

S'il n'est pas dans le `PATH`, indiquez son chemin avec l'option `--exiftool`.

### Modèles de reconnaissance

Au premier lancement, les modèles ONNX (environ 37 Mo) sont téléchargés automatiquement dans `~/.face_tagger/models` (sous Windows : `%USERPROFILE%\.face_tagger\models`).

## 🚀 Utilisation

```powershell
python face_tagger.py --directory "D:\Photos"
```

1. 📁 Choisissez le dossier de photos, puis cliquez sur **🔍 Analyser**. La première analyse d'un grand dossier peut être longue ; les suivantes ne traitent que les photos nouvelles ou modifiées.
2. 👥 La liste de gauche présente les personnes : celles déjà nommées dans les métadonnées, puis les inconnues. Les visages de la personne sélectionnée s'affichent à droite, page par page.
3. ✏️ Nommez les personnes : **Renommer…** (ou double-clic), **🧩 Nommer les inconnus…** pour les passer en revue une à une. Donner un nom déjà utilisé réunit les deux groupes.
4. 🧹 Corrigez au besoin : **🔗 Fusionner** plusieurs groupes, **🙈 Ignorer** les inconnus de passage, et clic droit sur un visage pour le déplacer, le détacher ou l'écarter. Le curseur **🎚️ Seuil de similarité** puis **🔄 Regrouper à nouveau** rendent les groupes plus larges (seuil bas) ou plus stricts (seuil haut).
5. 💾 Cochez les métadonnées à écrire, puis cliquez sur **Écrire dans les métadonnées**.

> 💡 Travaillez d'abord sur une **copie** de quelques photos pour vous familiariser avec l'outil, en laissant la sauvegarde `*_original` activée.

### Métadonnées écrites

| Case | Étiquette | Contenu |
|---|---|---|
| régions de visages (MWG) | `XMP-mwg-rs:RegionInfo` | Position et nom de chaque visage |
| PersonInImage | `XMP-iptcExt:PersonInImage` | Liste des personnes de la photo |
| mots-clés (XMP, EXIF) | `XMP-dc:Subject`, `XPKeywords` | Noms ajoutés aux mots-clés (`XPKeywords` est affiché par l'Explorateur Windows) |

Les étiquettes existantes sont conservées ; une personne renommée remplace seulement son ancien nom. Les mots-clés **IPTC** ne sont volontairement pas modifiés : leur jeu de caractères, pas toujours déclaré, risquerait d'abîmer les accents.

### Options de la ligne de commande

| Option | Défaut | Rôle |
|---|---|---|
| `--directory` | | Dossier de photos ouvert au démarrage |
| `--data-directory` | `~/.face_tagger` | Dossier des modèles et de la base |
| `--database` | `<data-directory>/face_tagger.sqlite` | Fichier de la base SQLite |
| `--page-size` | `100` | Nombre de vignettes par page |
| `--exiftool` | `exiftool` | Chemin ou nom de l'exécutable ExifTool |
| `--threshold` | `0.40` | Seuil de similarité (de 0.20 à 0.70) |
| `--max-image-size` | `2000` | Taille maximale (pixels) des photos pour la détection |
| `--min-face-size` | `40` | Taille minimale (pixels) d'un visage retenu |
| `--detection-score` | `0.80` | Score minimal de détection d'un visage (0 à 1) |
| `--no-backup` | | Ne pas conserver de copie `*_original` |
| `--no-emoji` | | Désactiver les emojis (si Tk plante à leur affichage) |
| `--theme` | `auto` | Thème : `auto` (celui du système), `light` ou `dark` |

Changer les paramètres de détection (`--max-image-size`, `--min-face-size`, `--detection-score`) entraîne une nouvelle analyse des visages ; les noms lus restent valables.

## ♻️ Sauvegardes et restauration

À la **première** modification d'une photo, ExifTool en garde une copie `photo.jpg_original`, qui n'est jamais remplacée ensuite : elle reste la version d'origine. Le bouton **♻️ Restaurer les sauvegardes…** remet ces copies en place. Le script `restore_originals.py` fait de même en ligne de commande :

```powershell
# Simulation : affiche ce qui serait fait, sans rien modifier
python restore_originals.py --directory "D:\Photos" --dry-run
# Restauration des versions d'origine
python restore_originals.py --directory "D:\Photos"
# Suppression des sauvegardes, une fois les modifications validées
python restore_originals.py --directory "D:\Photos" --action delete --yes
```

## 🗄️ Données de l'application

Tout est rangé dans `~/.face_tagger` (modifiable par `--data-directory`) :

- `models/` : les modèles ONNX ;
- `face_tagger.sqlite` : visages, empreintes, vignettes, étiquettes lues et historique des modifications.

Supprimer ce dossier remet l'application à zéro, sans toucher aux photos.

## 🛠️ Compilation

Les exécutables sont compilés par [GitHub Actions](.github/workflows/build.yml) avec PyInstaller, sous Windows, Linux et macOS, à la **publication d'une release** ou à la main depuis l'onglet *Actions*. Pour compiler en local :

```powershell
pip install pyinstaller
pyinstaller --onefile --noconsole --icon=face_tagger.ico --version-file=version.txt --add-data "face_tagger.ico;." face_tagger.py
```

L'exécutable est créé dans `dist/`. Les trois fichiers `.py` doivent rester dans le même dossier.

## 📂 Structure du dépôt

| Fichier | Rôle |
|---|---|
| `face_tagger.py` | Application (interface, détection, regroupement, écriture des métadonnées) |
| `face_database.py` | Base SQLite |
| `restore_originals.py` | Restauration ou suppression des sauvegardes `*_original` |
| `version.txt` | Informations de version de l'exécutable Windows |
| `face_tagger.ico`, `.png`, `.svg` | Icône |

## 📜 Licence

Distribué sous licence [MIT](LICENSE) — © PLAIS Lionel.

Modèles YuNet et SFace : [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Métadonnées : [ExifTool](https://exiftool.org/) de Phil Harvey.
