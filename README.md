# FaceTagger

Utilitaire de reconnaissance faciale pour photos

[![Build Multi-OS Executables](https://github.com/ILPlais/FaceTagger/actions/workflows/build.yml/badge.svg)](https://github.com/ILPlais/FaceTagger/actions/workflows/build.yml)

FaceTagger parcourt un dossier de photos (sous-dossiers compris), détecte les visages, les **regroupe par personne**, reprend les noms déjà présents dans les métadonnées des photos, vous propose de **nommer les personnes inconnues**, puis **écrit les noms dans les métadonnées** des photos, où les retrouvent l'Explorateur Windows et les logiciels de photos (digiKam, Lightroom…).

## ✨ Fonctionnalités

- 🔍 **Détection et reconnaissance locales** avec [DeepFace](https://github.com/serengil/deepface) : détecteur RetinaFace et modèle ArcFace par défaut, exécutés par onnxruntime, sans TensorFlow, sans GPU ni service en ligne : les photos ne quittent jamais l'ordinateur. D'autres détecteurs (YuNet, plus rapide) et modèles (Facenet512, SFace…) se choisissent en ligne de commande.
- 👥 **Regroupement par personne** selon la similarité des visages, avec un seuil réglable.
- 🏷️ **Reprise des noms existants** : régions de visages (MWG ou Microsoft), `PersonInImage` et, au choix, mots-clés.
- 🧩 **Nommage rapide des inconnus**, des plus fréquents aux plus rares ; renommage et fusion de groupes.
- ☑️ **Sélection de plusieurs vignettes** (Ctrl ou Maj + clic, Ctrl + A) pour les déplacer vers une autre personne, les détacher dans un nouveau groupe ou les écarter d'un coup.
- ✅ **Validation personne par personne** : le nom de la personne sélectionnée est écrit dans celles de ses photos où il manque encore, puis chaque photo est relue pour vérifier. La colonne « Métadonnées » indique les personnes déjà à jour.
- 🖼️ **Analyse d'un dossier entier ou de quelques photos** choisies une à une.
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

> ⚠️ DeepFace installe `opencv-contrib-python-headless`. Si l'environnement contient déjà `opencv-python` (version précédente de FaceTagger), désinstallez-le d'abord : les deux paquets entrent en conflit.
>
> ```powershell
> pip uninstall --yes opencv-python
> ```

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

Au premier lancement, DeepFace télécharge les poids de ses modèles (environ 255 Mo pour RetinaFace et ArcFace) dans `~/.face_tagger/.deepface/weights` (sous Windows : `%USERPROFILE%\.face_tagger\.deepface\weights`).

## 🚀 Utilisation

```powershell
python face_tagger.py --directory "D:\Photos"
```

1. 📁 Choisissez le dossier de photos, puis cliquez sur **🔍 Analyser**. La première analyse d'un grand dossier peut être longue ; les suivantes ne traitent que les photos nouvelles ou modifiées.
2. 👥 La liste de gauche présente les personnes : celles déjà nommées dans les métadonnées, puis les inconnues. Les visages de la personne sélectionnée s'affichent à droite, page par page.
3. ✏️ Nommez les personnes : **Renommer…** (ou double-clic), **🧩 Nommer les inconnus…** pour les passer en revue une à une. Donner un nom déjà utilisé réunit les deux groupes.
4. 🧹 Corrigez au besoin : **🔗 Fusionner** plusieurs groupes, **🙈 Ignorer** les inconnus de passage. Dans les vignettes, sélectionnez les visages mal attribués (clic, **Ctrl + clic** pour en ajouter, **Maj + clic** pour une plage, **Ctrl + A** pour tous), puis **Déplacer…**, **Détacher** ou **Écarter** (aussi par clic droit). Le curseur **🎚️ Seuil de similarité** puis **🔄 Regrouper à nouveau** rendent les groupes plus larges (seuil bas) ou plus stricts (seuil haut).
5. ✅ Cochez les métadonnées à écrire, puis validez les personnes **une à une** : sélectionnez-en une, vérifiez ses vignettes et cliquez sur **Valider la personne**. Seules les photos où son nom manque encore sont modifiées. **Tout écrire…** traite toutes les personnes nommées d'un coup, sans cette vérification.

Pour n'analyser que quelques photos, cliquez sur **🖼 Photos…** au lieu de choisir un dossier.

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
| `--model` | `ArcFace` | Modèle de reconnaissance de DeepFace : `ArcFace`, `Facenet512`, `GhostFaceNet`, `SFace`, `Facenet` ou `VGG-Face` |
| `--detector` | `retinaface` | Détecteur de visages : `retinaface`, `yunet` (environ dix fois plus rapide), `centerface`, `mtcnn`, `ssd` ou `opencv` |
| `--threshold` | selon le modèle (`0.32` pour ArcFace) | Seuil de similarité (de 0.20 à 0.80) |
| `--max-image-size` | `2000` | Taille maximale (pixels) des photos pour la détection |
| `--min-face-size` | `40` | Taille minimale (pixels) d'un visage retenu |
| `--detection-score` | `0.80` | Score minimal de détection d'un visage (0 à 1) |
| `--no-backup` | | Ne pas conserver de copie `*_original` |
| `--no-emoji` | | Désactiver les emojis (si Tk plante à leur affichage) |
| `--theme` | `auto` | Thème : `auto` (celui du système), `light` ou `dark` |

Changer de modèle ou de détecteur, ou les paramètres de détection (`--max-image-size`, `--min-face-size`, `--detection-score`), entraîne une nouvelle analyse des visages ; les noms lus restent valables. Chaque modèle a sa propre échelle de ressemblance : sans `--threshold`, le seuil conseillé par DeepFace pour ce modèle est utilisé.

> ⏱️ Sur le processeur, RetinaFace prend environ 3,5 s par photo, YuNet environ 0,35 s. Pour une grande photothèque, `--detector yunet` accélère beaucoup la première analyse.

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

- `.deepface/weights/` : les poids des modèles de DeepFace ;
- `face_tagger.sqlite` : visages, empreintes, vignettes, étiquettes lues et historique des modifications.

Supprimer ce dossier remet l'application à zéro, sans toucher aux photos.

## 🛠️ Compilation

Les exécutables sont compilés par [GitHub Actions](.github/workflows/build.yml) avec PyInstaller, sous Windows, Linux et macOS, à la **publication d'une release** ou à la main depuis l'onglet *Actions*. Pour compiler en local :

```powershell
pip install pyinstaller
pyinstaller --onefile --noconsole --icon=face_tagger.ico --version-file=version.txt --add-data "face_tagger.ico;." --collect-submodules deepface.models --collect-submodules deepface.modules --recursive-copy-metadata deepface face_tagger.py
```

Les options `--collect-submodules` et `--recursive-copy-metadata` sont nécessaires : DeepFace charge ses modèles par leur nom, ce que PyInstaller ne détecte pas seul.

L'exécutable est créé dans `dist/`. Les trois fichiers `.py` doivent rester dans le même dossier.

## 📂 Structure du dépôt

| Fichier | Rôle |
|---|---|
| `face_tagger.py` | Application (interface, détection, regroupement, écriture des métadonnées) |
| `face_database.py` | Base SQLite |
| `restore_originals.py` | Restauration ou suppression des sauvegardes `*_original` |
| `version.txt` | Informations de version de l'exécutable Windows |
| `face_tagger.ico`, `.png`, `.svg` | Icône |

## 💖 Soutenir le projet

FaceTagger est gratuit et le restera. Si l'outil vous rend service, vous pouvez soutenir son développement :

| ₿ Bitcoin | Ξ Ethereum | 💳 PayPal |
|:---:|:---:|:---:|
| <img src=".github/funding/bitcoin.png" alt="QR-code Bitcoin" width="160"> | <img src=".github/funding/ethereum.png" alt="QR-code Ethereum" width="160"> | <img src=".github/funding/paypal.png" alt="QR-code PayPal" width="160"> |
| `bc1qd8875ezcdzmncl2pe3cek2ymupje5fhddynw36` | `0x771B29bc8dCCdbFa33e0Ad65ec507bdF3fcf05A5` | [paypal.me/LionelPlais](https://paypal.me/LionelPlais) |

Merci ! 🙏

## 📜 Licence

Distribué sous licence [MIT](LICENSE) — © PLAIS Lionel.

Détection et reconnaissance des visages : [DeepFace](https://github.com/serengil/deepface) de Sefik Ilkin Serengil (licence MIT). Métadonnées : [ExifTool](https://exiftool.org/) de Phil Harvey.
