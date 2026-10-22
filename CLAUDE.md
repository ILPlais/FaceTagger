# FaceTagger — instructions pour Claude Code

Utilitaire Python à interface graphique (Tkinter) : il parcourt un dossier de photos (sous-dossiers compris), détecte les visages, les **regroupe par personne**, reprend les noms déjà présents dans les métadonnées des photos, propose de **nommer les personnes inconnues**, puis **écrit les noms dans les métadonnées** (via ExifTool).

- Cible principale : **Windows** (binaires Linux et macOS aussi produits par GitHub Actions).
- Dépôt : https://github.com/ILPlais/FaceTagger — licence MIT — auteur : PLAIS Lionel.
- Langue : tout est en **français** (interface, messages, commentaires, docstrings, commits, documentation).

## Conventions de code (à respecter strictement)

- Indentation avec des **tabulations**, jamais des espaces. Les alignements de continuation se font aussi avec des tabulations seules.
- Python : **espaces de chaque côté du `=`** dans les appels de fonctions et les arguments nommés.

	```python
	# Correct
	parser.add_argument("--threshold", type = float, default = 0.40)
	# Incorrect
	parser.add_argument("--threshold", type=float, default=0.40)
	```

- Options de ligne de commande en **noms longs** (`--directory`, jamais `-d`), y compris dans les commandes documentées (`pip install --requirement requirements.txt`, `git commit --message`).
- Messages affichés par les programmes (console, journal, boîtes de dialogue, boutons) : **avec des emojis** (✅ ❌ ⚠️ 🔍 …). L'option `--no-emoji` doit continuer à fonctionner : tout texte passe par `emoji()` (ou est traité par `strip_widget_emoji()` pour les widgets).
- Code **bien commenté** : docstrings et commentaires en français, qui expliquent le *pourquoi* et pas seulement le *quoi*.
- Messages de commit en français, dans le style existant (« Ajoute … », « Suppression de … »). **Ne jamais commiter ni pousser sans demande explicite.**
- Rester compatible **Python 3.11** (version utilisée par la CI) même si le `.venv` local est en 3.14 : pas de syntaxe plus récente. Vérifié : les trois fichiers compilent en 3.11.

## Installation et lancement (Windows, PowerShell)

```powershell
# Création de l'environnement virtuel puis installation des dépendances Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --requirement requirements.txt

# ExifTool : programme externe indispensable (lecture et écriture des métadonnées)
winget install --id OliverBetz.ExifTool --exact

# Lancement de l'application
python face_tagger.py --directory "D:\Photos"
```

- Tkinter n'est pas sur PyPI : il est fourni par l'installeur de python.org (case « tcl/tk and IDLE »). Contrôle : `python -m tkinter`.
- Ne jamais installer `opencv-python` et `opencv-python-headless` ensemble.
- Versions dans le `.venv` actuel : Python 3.14, OpenCV 5.0.0.93, NumPy 2.5.4, Pillow 13.0.0, pillow-heif 1.8.1, PyInstaller 6.22.3. (`ttkthemes` y est aussi installé mais n'est ni utilisé ni dans `requirements.txt`.)
- Au premier lancement, les modèles ONNX (≈ 37 Mo) sont téléchargés dans `%USERPROFILE%\.face_tagger\models`.

Options de `face_tagger.py` : `--directory`, `--data-directory` (défaut `~/.face_tagger`), `--database` (défaut `<data-directory>/face_tagger.sqlite`), `--page-size` (100), `--exiftool`, `--threshold` (0.40, borné entre 0.20 et 0.70), `--max-image-size` (2000), `--min-face-size` (40), `--detection-score` (0.80), `--no-backup`, `--no-emoji`.

Restauration des sauvegardes `*_original` d'ExifTool (aussi disponible par le bouton « Restaurer les sauvegardes… ») :

```powershell
# Simulation, puis restauration réelle, puis suppression des sauvegardes
python restore_originals.py --directory "D:\Photos" --dry-run
python restore_originals.py --directory "D:\Photos"
python restore_originals.py --directory "D:\Photos" --action delete --yes
```

## Compilation

- **En local** : `pyinstaller face_tagger.spec` (onefile, `console = False`, icône, `version.txt`, `face_tagger.ico` en données). Le `.spec` est ignoré par git (`*.spec` dans `.gitignore`) : il n'existe que sur le poste de Lionel.
- **GitHub Actions** (`.github/workflows/build.yml`) : déclenché par la **publication d'une release** (`release: published`, Lionel les crée depuis la page Releases du site) ou **à la main** depuis l'onglet Actions (`workflow_dispatch`, en saisissant le tag). Matrice ubuntu / macos / windows, Python 3.11. Il appelle `pyinstaller --onefile` directement (sans le `.spec`) et publie `dist/*` dans la Release.
	- L'ancien déclencheur `push: tags` ne partait pas, et aucun nouveau tag n'existait sur GitHub. Causes probables (non vérifiées, faute d'accès à l'onglet Actions) : une release en **brouillon** ne crée pas son tag, et une release qui reprend un tag existant ne pousse aucun tag. Une release brouillon ne déclenche aucun workflow : il faut la **publier**.
	- GitHub lit le workflow **dans le commit du tag** : les tags `v1.0.0` et `v1.0.4` (posés sur `04d0e7c`) ont l'ancien déclencheur. Pour les recompiler, utiliser le lancement manuel.
	- Le dépôt est **privé** : l'API GitHub répond 404 sans authentification, et `gh` n'est pas installé sur le poste.
- Les trois fichiers `.py` restent **dans le même dossier** : PyInstaller embarque `face_database` et `restore_originals` grâce aux imports (vérifié par un build réel).
- `version.txt` (infos de version Windows) est à **1.0.0.0** alors que le dernier tag est **v1.0.4** : à resynchroniser à chaque publication.
- En mode fenêtré (`--noconsole`), `sys.stdout` et `sys.stderr` valent `None` : `print()` reste sans effet (pas de plantage), mais les `sys.exit("message")` du démarrage sont invisibles. Amélioration possible : les afficher dans une `messagebox`.
- Icône : `iconbitmap()` avec un `.ico` lève une `TclError` sous Linux/macOS, l'appel est donc protégé par `try/except`. `resource_path()` lit `sys._MEIPASS` sous PyInstaller, sinon le **dossier courant** (l'icône n'est trouvée en développement que si on lance depuis le dossier du dépôt).

## Architecture

| Fichier | Rôle |
|---|---|
| `face_tagger.py` | Application complète (≈ 1750 lignes, un seul fichier découpé en sections par des bandeaux `# ====`). |
| `face_database.py` | Classe `Database` : base SQLite (bibliothèque standard uniquement). |
| `restore_originals.py` | Restauration / suppression des `*_original` : ligne de commande **et** fonctions réutilisées par l'interface. |
| `version.txt`, `face_tagger.ico/.png/.svg`, `requirements.txt`, `README.md` | Ressources, dépendances et documentation (le README est très succinct). |

Sections de `face_tagger.py`, dans l'ordre :

1. **Constantes** : extensions, URL des modèles, `TAGS_TO_READ` (étiquettes lues par ExifTool).
2. **Utilitaires** : `emoji`, `resource_path`, `normalize_name`, `name_key`, `merge_unique`, `iou`, `list_photos`, `make_thumbnail`, `encode_thumbnail` / `decode_thumbnail`, `ensure_models`.
3. **ExifTool** : `PhotoTags`, `parse_photo_tags`, classe `ExifTool` (`read_tags`, `write_tags`).
4. **Détection et reconnaissance** : `FaceEngine` (YuNet pour la détection, SFace pour l'empreinte à 128 dimensions, via OpenCV).
5. **Modèle de données** : `FaceRecord`, `Person`, `cluster_faces`, `Library` (`label_faces`, `rebuild`, `rename`, `merge`, `move_face`, `delete_face`).
6. **Métadonnées à écrire** : `build_region_info`, `check_written`.
7. **Interface** : `NameDialog`, `PhotoViewer`, `UnknownNamesDialog`, `FaceTaggerApp`.
8. **Ligne de commande** : `parse_arguments`, `main`.

### Flux de données

- **Analyse** (thread, `scan_worker`) : `list_photos` → signatures en base → `FaceEngine.analyse` seulement pour les photos nouvelles ou modifiées → `Database.store_analysis` → étiquettes lues en base, sinon par ExifTool → `Library.label_faces` → `Library.rebuild` → événement `library` vers l'interface.
- **Écriture** (thread, `write_worker`) : `collect_updates` → un enregistrement par photo, fusionné avec les étiquettes existantes → `ExifTool.write_tags` par lots de 25 → relecture → `check_written` → mise à jour de la base (signature, étiquettes, historique).

## Décisions et pièges à connaître

### ExifTool

- Les arguments (noms de fichiers compris) sont envoyés en **UTF-8 par l'entrée standard** (`-@ -`, `-charset filename=utf8`) : indispensable pour les accents sous Windows et pour la longueur de la ligne de commande.
- L'écriture passe par un fichier JSON (`-json=records.json`) dans un **dossier temporaire utilisé comme répertoire de travail**, avec un chemin relatif : aucun chemin à accents (nom d'utilisateur Windows) n'est passé en argument. `CREATE_NO_WINDOW` évite les fenêtres console.
- Champs écrits : `XMP-iptcExt:PersonInImage`, `XMP-dc:Subject`, `XPKeywords` (EXIF, visible dans l'Explorateur Windows) et `XMP-mwg-rs:RegionInfo`. **On n'écrit volontairement pas `IPTC:Keywords`** : son jeu de caractères n'est pas toujours déclaré, risque de corrompre les accents.
- Régions **MWG** : `X` / `Y` désignent le **centre** du rectangle, valeurs normalisées. Régions **Microsoft** (`RegionInfoMP`) : coin supérieur gauche. Les coordonnées des visages sont normalisées sur l'image **redressée selon l'orientation EXIF** (`ImageOps.exif_transpose`).
- `*_original` : ExifTool le crée à la **première** écriture et ne l'écrase plus ensuite (vérifié) : il contient toujours la version d'origine. Avec `--no-backup`, on passe `-overwrite_original`.
- Une personne renommée **remplace** son ancien nom dans les régions qui la recouvrent (IoU ≥ 0.15) et dans les listes (`PersonInImage`, `Subject`, `XPKeywords`). Après chaque lot, les photos sont relues pour vérifier que les noms y sont.

### Regroupement des visages

- Similarité cosinus, seuil 0.40 par défaut (curseur de 0.20 à 0.70).
- Les visages déjà étiquetés (régions) sont regroupés **par nom d'abord** ; deux groupes portant des noms différents ne fusionnent **jamais**.
- Les noms sans position (`PersonInImage`, et mots-clés si la case est cochée) sont attribués à un groupe seulement s'ils figurent dans ≥ 60 % de ses photos et dominent nettement (le 2ᵉ candidat ≤ la moitié du 1ᵉʳ) ; un nom n'est donné qu'à un seul groupe.

### Base SQLite (`~/.face_tagger/face_tagger.sqlite`)

- Mode WAL, **une connexion par thread** (`threading.local`) : ne jamais partager une connexion entre threads.
- Tables : `meta`, `photos` (signature = `mtime_ns` + taille, `tags_json` = JSON brut d'ExifTool), `faces` (position, empreinte `float32` × 128 en BLOB, vignette JPEG), `modifications` (historique, `backup_path`, `restored_at`).
- Les vignettes ne sont chargées **qu'à la demande** (cache LRU de 3000). Une photo inchangée n'est ni ré-analysée ni relue.
- `reset_analysis_if_needed` invalide les visages si les paramètres de détection changent (les étiquettes restent valables). Après une écriture, la signature est mise à jour (les pixels n'ont pas changé : les visages restent valables) ; après une restauration, les étiquettes mémorisées sont oubliées.
- Si le schéma change : incrémenter `SCHEMA_VERSION` **et** écrire une migration.

### Interface

- Les threads de travail **ne touchent jamais Tk** : ils envoient des événements avec `self.post(...)` (file `queue.Queue`), lus par `poll_events` toutes les 100 ms.
- Conserver une référence à chaque `ImageTk.PhotoImage` (`icon_cache`, `face_images`), sinon l'image disparaît.
- Affichage **paginé** (`render_page`, 50 / 100 / 200 / 500 par page) : seules les vignettes de la page courante sont en mémoire.

### Modèles et OpenCV

- YuNet (`face_detection_yunet_2023mar.onnx`) et SFace (`face_recognition_sface_2021dec.onnx`), du dépôt `opencv_zoo`, téléchargés dans `~/.face_tagger/models`. Sur GitHub ces fichiers sont en **Git LFS** (redirection vers `media.githubusercontent.com`) : `MODEL_MIN_SIZE` détecte un simple pointeur LFS.
- L'API `FaceDetectorYN` / `FaceRecognizerSF` est identique en OpenCV 4.x et 5.0 (vérifié). **OpenCV 5.0 n'a plus `CascadeClassifier`** : ne pas l'utiliser, notamment dans des tests.

## Tests

**Il n'y a pas encore de tests automatisés dans le dépôt.** Lors du développement (sous Linux, avec un faux moteur de détection) ont été vérifiés : regroupement, attribution des noms, écriture et relecture des métadonnées (accents inclus), idempotence de l'écriture, base SQLite, restauration, pagination, flux complet de l'interface, build PyInstaller. Les **vrais modèles ONNX n'ont pas été exécutés** en dehors de l'ordinateur de Lionel.

Pour en ajouter (pytest, dossier `tests/`) :

- Fonctions pures : `cluster_faces` (empreintes synthétiques), `parse_photo_tags`, `build_region_info`, `check_written`, `iou`.
- `Database` et `restore_originals` : sur des **copies** de photos, avec ExifTool.
- Interface : remplacer `face_tagger.FaceEngine` et `face_tagger.ensure_models` par un faux moteur renvoyant des empreintes synthétiques (octets `float32` × 128) et des vignettes JPEG ; remplacer `messagebox.askyesno` ; piloter `start_scan`, `start_write`, `start_restore` ; sous Linux, utiliser `xvfb-run`.
- **Ne jamais tester sur de vraies photos** : toujours des copies, en gardant les `*_original`.

## Points à traiter

- Ajouter des tests et compléter le `README.md` (installation, usage, captures).
- Supprimer la constante `MAX_FACES_SHOWN` (inutilisée depuis la pagination) et le fichier obsolète `~/.face_tagger/faces_cache.pkl` (ancien cache de la v1).
- Resynchroniser `version.txt` avec le tag ; afficher dans une `messagebox` les erreurs de démarrage en mode fenêtré.
- Tester les vrais modèles sur un petit dossier de photos (copies), avec la sauvegarde `*_original` activée.
