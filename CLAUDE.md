# FaceTagger — instructions pour Claude Code

Utilitaire Python à interface graphique (Tkinter) : il parcourt un dossier de photos (sous-dossiers compris) ou quelques photos choisies, détecte les visages avec **DeepFace**, les **regroupe par personne**, reprend les noms déjà présents dans les métadonnées des photos, propose de **nommer les personnes inconnues**, puis **écrit les noms dans les métadonnées** (via ExifTool), **une personne à la fois** après vérification de ses vignettes.

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
- **Docstrings au style Google** : résumé d'une ligne sur la ligne des `"""`, explication éventuelle, puis les sections `Args:` (`nom (type): description`, `*args` compris), `Returns:` (`type: description`, omise pour `-> None`), `Raises:` (exceptions levées volontairement) et, pour les classes et dataclasses, `Attributes:` (chaque attribut avec son type). Les propriétés commencent par leur type (`bool: True si …`). Titres de section en **anglais** (seuls reconnus par Sphinx/napoleon), contenu en français ; types identiques aux annotations ; continuations indentées d'une tabulation de plus. Les docstrings de module ont aussi leur résumé en première ligne, et leurs exemples sous `Example:` / `Examples:`.
- **Docstring** pour chaque module, classe, fonction et méthode (y compris les fonctions imbriquées) ; **annotations de type** pour tous les paramètres et retours (`-> None` compris, `__init__` aussi), les constantes de module et chaque attribut d'instance à sa première affectation (`self.page: int = 0`). Variables locales typées quand le type ne se lit pas (conteneur vide, `None` initial, `json.loads`). Les trois fichiers ont `from __future__ import annotations` : les annotations ne sont pas évaluées à l'exécution, mais doivent se résoudre (vérifié avec `typing.get_type_hints`). Le module DeepFace est typé `ModuleType`, la base de `restore_originals` `Optional[Database]` (import sous `TYPE_CHECKING`).
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
- Ne jamais installer `opencv-python` et `opencv-python-headless` (ou `opencv-contrib-python-headless`) ensemble. DeepFace dépend de `opencv-contrib-python-headless` : `opencv-python` a été désinstallé du `.venv`.
- Versions dans le `.venv` actuel : Python 3.14, DeepFace 0.0.105, onnxruntime 1.32.0, opencv-contrib-python-headless 5.1.0.95, NumPy 2.5.4, Pillow 13.0.0, pillow-heif 1.8.1, PyInstaller 6.22.3. **Ni TensorFlow ni PyTorch** : DeepFace 0.0.105 s'exécute avec onnxruntime. (`ttkthemes` y est aussi installé mais n'est ni utilisé ni dans `requirements.txt`.)
- Au premier lancement, DeepFace télécharge les poids (≈ 255 Mo : `arcface_weights.onnx` 137 Mo, `retinaface.onnx` 118 Mo) dans `%USERPROFILE%\.face_tagger\.deepface\weights`. L'ancien dossier `models` (YuNet, SFace) ne sert plus.

Options de `face_tagger.py` : `--directory`, `--data-directory` (défaut `~/.face_tagger`), `--database` (défaut `<data-directory>/face_tagger.sqlite`), `--page-size` (100), `--exiftool`, `--model` (`ArcFace` par défaut, voir `RECOGNITION_MODELS`), `--detector` (`retinaface` par défaut, voir `DETECTOR_BACKENDS`), `--threshold` (défaut : celui du modèle, 0.32 pour ArcFace ; borné entre 0.20 et 0.80), `--max-image-size` (2000), `--min-face-size` (40), `--detection-score` (0.80), `--no-backup`, `--no-emoji`, `--theme` (`auto`, `light` ou `dark` ; défaut `auto`, qui suit Windows).

Restauration des sauvegardes `*_original` d'ExifTool (aussi disponible par le bouton « Restaurer les sauvegardes… ») :

```powershell
# Simulation, puis restauration réelle, puis suppression des sauvegardes
python restore_originals.py --directory "D:\Photos" --dry-run
python restore_originals.py --directory "D:\Photos"
python restore_originals.py --directory "D:\Photos" --action delete --yes
```

## Compilation

- **En local** : `pyinstaller face_tagger.spec` (onefile, `console = False`, icône, `version.txt`, `face_tagger.ico` en données). Le `.spec` est ignoré par git (`*.spec` dans `.gitignore`) : il n'existe que sur le poste de Lionel.
- **DeepFace sous PyInstaller** : il faut `--collect-submodules deepface.models --collect-submodules deepface.modules` (DeepFace importe ses détecteurs et modèles par leur nom, avec `importlib`) et `--recursive-copy-metadata deepface` (sinon « No package metadata was found for gdown »). Ajoutés au workflow et au `.spec` local ; vérifié par la compilation réelle (onedir, console) d'un petit programme qui importe `face_tagger` et analyse une image. Le build en `--onefile` fenêtré complet n'a pas été refait.
- **GitHub Actions** (`.github/workflows/build.yml`) : déclenché par la **publication d'une release** (`release: published`, Lionel les crée depuis la page Releases du site) ou **à la main** depuis l'onglet Actions (`workflow_dispatch`, en saisissant le tag). Matrice `ubuntu-latest`, `ubuntu-24.04-arm`, `macos-latest` (Apple Silicon, ARM64), `windows-latest` et `windows-11-arm`, Python 3.11, `fail-fast: false`. Windows ARM64 est expérimental (`continue-on-error`) : toutes les dépendances (OpenCV notamment) n'y ont peut-être pas de paquet précompilé (non vérifié). Les runners ARM de Linux et Windows sont disponibles dans les dépôts privés depuis janvier 2026. Pas de macOS Intel (`macos-15-intel`, dernier runner x86_64, maintenu jusqu'en août 2027). Il appelle `pyinstaller --onefile` directement (sans le `.spec`), puis place l'exécutable dans une archive nommée d'après le tag, le système et l'architecture (`face_tagger-v1.2.3-linux-x86_64.tar.gz`, `…-macos-arm64.tar.gz`, `…-windows-x86_64.zip`), publiée dans la Release. Sans cela, les exécutables Linux et macOS portaient le même nom (`face_tagger`) et le dernier envoyé écrasait l'autre.
	- L'ancien déclencheur `push: tags` ne partait pas, et aucun nouveau tag n'existait sur GitHub. Causes probables (non vérifiées, faute d'accès à l'onglet Actions) : une release en **brouillon** ne crée pas son tag, et une release qui reprend un tag existant ne pousse aucun tag. Une release brouillon ne déclenche aucun workflow : il faut la **publier**.
	- GitHub lit le workflow **dans le commit du tag** : les tags `v1.0.0` et `v1.0.4` (posés sur `04d0e7c`) ont l'ancien déclencheur. Pour les recompiler, utiliser le lancement manuel.
	- Le dépôt est **privé** : l'API GitHub répond 404 sans authentification, et `gh` n'est pas installé sur le poste.
- Les trois fichiers `.py` restent **dans le même dossier** : PyInstaller embarque `face_database` et `restore_originals` grâce aux imports (vérifié par un build réel).
- `version.txt` (infos de version Windows) est **mis à jour par le workflow** d'après le tag (`v1.2.3` → `1.2.3.0`, `v1.2.3.4` → `1.2.3.4`) ; un tag d'une autre forme (`v2.0-beta`) arrête tout le workflow.
	- Le job `version` le commite et le pousse sur la branche principale (auteur `github-actions[bot]`, message « Version 1.2.3.0 dans version.txt (release v1.2.3). »), seulement si le tag est **plus récent** que le fichier : recompiler un ancien tag ne fait pas reculer la version. Un échec du push (branche protégée…) laisse un avertissement sans bloquer la compilation. **Penser à `git pull` après une release.**
	- Le commit du tag étant antérieur à ce commit, la compilation Windows réécrit aussi `version.txt` dans la copie du runner.
	- Inutile donc de le resynchroniser à la main avant de poser le tag.
- En mode fenêtré (`--noconsole`), `sys.stdout` et `sys.stderr` valent `None` : `main()` les remplace par `os.devnull`, car les barres de progression des téléchargements de DeepFace (tqdm) y écrivent. Les `sys.exit("message")` du démarrage restent invisibles. Amélioration possible : les afficher dans une `messagebox`.
- Icône : `iconbitmap(default = …)` (les boîtes de dialogue en héritent) ; avec un `.ico`, l'appel lève une `TclError` sous Linux/macOS et est donc protégé par `try/except`. `resource_path()` lit `sys._MEIPASS` sous PyInstaller, sinon le **dossier du script** : l'ancienne version cherchait dans le dossier courant, et une fois lancée hors du dépôt, l'application montrait la plume de Tk dans la barre des tâches (vérifié par capture d'écran).
- Barre des tâches de Windows : `set_windows_app_id()` déclare l'`AppUserModelID` `ILPlais.FaceTagger` avant la création de la fenêtre. Sinon, lancée par `python face_tagger.py`, la fenêtre est rattachée à python.exe, et en mode « regrouper les boutons » Windows affiche l'icône de Python (comportement documenté, non reproduit : la barre des tâches de Lionel ne regroupe jamais).

## Architecture

| Fichier | Rôle |
|---|---|
| `face_tagger.py` | Application complète (≈ 4000 lignes, docstrings comprises, un seul fichier découpé en sections par des bandeaux `# ====`). |
| `face_database.py` | Classe `Database` : base SQLite (bibliothèque standard uniquement). |
| `restore_originals.py` | Restauration / suppression des `*_original` : ligne de commande **et** fonctions réutilisées par l'interface. |
| `version.txt`, `face_tagger.ico/.png/.svg`, `requirements.txt`, `README.md` | Ressources, dépendances et documentation (le README est très succinct). |

Sections de `face_tagger.py`, dans l'ordre :

1. **Constantes** : extensions, URL des modèles, `TAGS_TO_READ` (étiquettes lues par ExifTool).
2. **Utilitaires** : `emoji`, `resource_path`, `normalize_name`, `name_key`, `merge_unique`, `iou`, `list_photos`, `make_thumbnail`, `encode_thumbnail` / `decode_thumbnail`, `load_deepface`.
3. **ExifTool** : `PhotoTags`, `parse_photo_tags`, classe `ExifTool` (`read_tags`, `write_tags`).
4. **Détection et reconnaissance** : `FaceEngine` (DeepFace : `build_model` à la création, puis `represent` par photo, qui détecte, aligne et calcule l'empreinte).
5. **Modèle de données** : `FaceRecord`, `Person`, `cluster_faces`, `Library` (`label_faces`, `rebuild`, `rename`, `merge`, `move_face`, `delete_face`).
6. **Métadonnées à écrire** : `build_region_info`, `check_written`, `photos_missing_name`.
7. **Interface** : `ToolTip`, `photo_caption`, `NameDialog`, `PhotoViewer`, `UnknownNamesDialog`, `FaceTaggerApp`.
8. **Ligne de commande** : `parse_arguments`, `main`.

### Flux de données

- **Analyse** (thread, `scan_worker`) : `list_photos` (ou les photos choisies par « Photos… ») → signatures en base → `FaceEngine.analyse` seulement pour les photos nouvelles ou modifiées → `Database.store_analysis` → étiquettes lues en base, sinon par ExifTool → `Library.label_faces` → `Library.rebuild` → événement `library` vers l'interface.
- **Écriture** (thread, `write_worker`) : `validate_person` (une personne) ou `start_write` (toutes) → `collect_updates(persons)`, qui ne retient que les photos où le nom **manque encore** (`photos_missing_name`) → `launch_write` → un enregistrement par photo, fusionné avec les étiquettes existantes → `ExifTool.write_tags` par lots de 25 → relecture → `check_written` → mise à jour de la base (signature, étiquettes, historique) → événement `tags`, qui met à jour la colonne « Métadonnées ».

## Décisions et pièges à connaître

### ExifTool

- Les arguments (noms de fichiers compris) sont envoyés en **UTF-8 par l'entrée standard** (`-@ -`, `-charset filename=utf8`) : indispensable pour les accents sous Windows et pour la longueur de la ligne de commande.
- L'écriture passe par un fichier JSON (`-json=records.json`) dans un **dossier temporaire utilisé comme répertoire de travail**, avec un chemin relatif : aucun chemin à accents (nom d'utilisateur Windows) n'est passé en argument. `CREATE_NO_WINDOW` évite les fenêtres console.
- Champs écrits : `XMP-iptcExt:PersonInImage`, `XMP-dc:Subject`, `XPKeywords` (EXIF, visible dans l'Explorateur Windows) et `XMP-mwg-rs:RegionInfo`. **On n'écrit volontairement pas `IPTC:Keywords`** : son jeu de caractères n'est pas toujours déclaré, risque de corrompre les accents.
- Régions **MWG** : `X` / `Y` désignent le **centre** du rectangle, valeurs normalisées. Régions **Microsoft** (`RegionInfoMP`) : coin supérieur gauche. Les coordonnées des visages sont normalisées sur l'image **redressée selon l'orientation EXIF** (`ImageOps.exif_transpose`).
- `*_original` : ExifTool le crée à la **première** écriture et ne l'écrase plus ensuite (vérifié) : il contient toujours la version d'origine. Avec `--no-backup`, on passe `-overwrite_original`.
- Une personne renommée **remplace** son ancien nom dans les régions qui la recouvrent (IoU ≥ 0.15) et dans les listes (`PersonInImage`, `Subject`, `XPKeywords`). Après chaque lot, les photos sont relues pour vérifier que les noms y sont.

### Regroupement des visages

- Similarité cosinus. Chaque modèle a son échelle : le seuil par défaut est celui de DeepFace pour ce modèle (`deepface/config/threshold.py`, distance cosinus convertie en similarité), dans `RECOGNITION_MODELS` : ArcFace 0.32, Facenet512 0.70, SFace 0.41… Curseur de 0.20 à 0.80.
- Sur les images d'exemple de DeepFace (`tests/unit/dataset`), ArcFace + RetinaFace au seuil 0.32 regroupe bien les 8 visages d'une même personne. Vitesse mesurée sur le processeur : RetinaFace ≈ 3,5 s par photo, YuNet ≈ 0,35 s (mêmes visages trouvés), CenterFace ≈ 3 s.
- Les visages déjà étiquetés (régions) sont regroupés **par nom d'abord** ; deux groupes portant des noms différents ne fusionnent **jamais**.
- Les noms sans position (`PersonInImage`, et mots-clés si la case est cochée) sont attribués à un groupe seulement s'ils figurent dans ≥ 60 % de ses photos et dominent nettement (le 2ᵉ candidat ≤ la moitié du 1ᵉʳ) ; un nom n'est donné qu'à un seul groupe.

### Base SQLite (`~/.face_tagger/face_tagger.sqlite`)

- Mode WAL, **une connexion par thread** (`threading.local`) : ne jamais partager une connexion entre threads.
- Tables : `meta`, `photos` (signature = `mtime_ns` + taille, `tags_json` = JSON brut d'ExifTool), `faces` (position, empreinte `float32` × 128 en BLOB, vignette JPEG), `modifications` (historique, `backup_path`, `restored_at`).
- Les vignettes ne sont chargées **qu'à la demande** (cache LRU de 3000). Une photo inchangée n'est ni ré-analysée ni relue.
- `reset_analysis_if_needed` invalide les visages si le moteur ou les paramètres de détection changent (signature `deepface|moteur|modèle|détecteur|taille|visage min|score` ; les étiquettes restent valables). Les empreintes de modèles différents ne sont pas comparables, et leur dimension varie (128 à 4096) : le BLOB `embedding` n'a pas de taille fixe. Après une écriture, la signature est mise à jour (les pixels n'ont pas changé : les visages restent valables) ; après une restauration, les étiquettes mémorisées sont oubliées.
- Si le schéma change : incrémenter `SCHEMA_VERSION` **et** écrire une migration.

### Interface

- Les threads de travail **ne touchent jamais Tk** : ils envoient des événements avec `self.post(...)` (file `queue.Queue`), lus par `poll_events` toutes les 100 ms.
- Conserver une référence à chaque `ImageTk.PhotoImage` (`icon_cache`, `face_images`), sinon l'image disparaît.
- Affichage **paginé** (`render_page`, 50 / 100 / 200 / 500 par page) : seules les vignettes de la page courante sont en mémoire.
- **Sélection de plusieurs vignettes** : `selected_faces` (ensemble de `FaceRecord`, comparés par identité), `selection_anchor` (rang dans `sorted_faces`, pour Maj + clic, pages précédentes comprises), `face_widgets` (vignettes de la page). Clic, Ctrl + clic, Maj + clic sur une vignette ; Ctrl + A, Échap et Suppr sont liés au **canevas**, qui prend le focus à chaque clic. La sélection est vidée quand la personne affichée change. Les vignettes sélectionnées prennent le style `SelectedFace.TLabel` (couleurs dans `SELECTION_COLORS`). Le menu contextuel agit sur toute la sélection si la vignette cliquée en fait partie.
- **Validation par personne** : le bouton « Valider la personne » n'est actif que pour une seule personne nommée et non ignorée (`update_action_states`, appelée aussi par `set_busy`). La colonne « Métadonnées » (`person_status`) dépend des cases « Écrire » : elle est recalculée par `update_person_statuses` (sans reconstruire la liste) à chaque changement de case et après chaque écriture. « Tout écrire… » reste disponible.
- **Cadre des vignettes** : vert si le visage est **déjà marqué** (une région de la photo qui le recouvre, IoU ≥ 0.15, porte déjà le nom de la personne), bleu s'il est seulement **proposé** (`face_is_tagged`, couleurs dans `FACE_FRAME_COLORS`). Le cadre est dessiné **dans l'image** par `thumbnail_photo(…, border = …)` (les thèmes natifs de ttk ignorent la couleur de bordure, et le fond du widget sert à la sélection). L'état est calculé d'après `library.photo_tags` et non `FaceRecord.label` (figé à l'analyse) : après une écriture, l'événement `tags` appelle `update_face_frames`, qui redessine les cadres de la page sans la reconstruire. Le titre de la personne donne le nombre de visages marqués et proposés, et son infobulle la légende.
- **Aperçu d'une photo** (`PhotoViewer`, double-clic sur une vignette) : cadre rouge pour le visage sélectionné, vert pour les autres ; un visage déjà enregistré (`face_is_tagged`) porte en plus 💾 dans le coin supérieur droit de son cadre, avec une légende sous la photo. Le badge (`saved_badge` : fond blanc arrondi et disquette) est **translucide** pour laisser voir le visage : opacités dans `SAVED_BADGE_OPACITY` (fond 0.35, disquette 0.70), appliquées au canal alpha puis collées avec ce canal comme masque. L'emoji est dessiné par Pillow (`embedded_color`) avec la première police de `EMOJI_FONTS` trouvée (Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji ; les polices à images intégrées n'acceptent que leurs tailles fixes, d'où la taille associée) ; sans police, ou avec `--no-emoji`, `floppy_pictogram` dessine une disquette. Vérifié sous Linux (Noto Color Emoji), **pas encore sous Windows** (Segoe UI Emoji).
- **Photos en erreur** (`Library.photo_errors`, chemin → description) : cadre **rouge** (`FACE_FRAME_COLORS["error"]`, prioritaire sur le vert et le bleu), état dans l'infobulle et le titre (🟥), description en rouge **au-dessus** de la photo dans `PhotoViewer`. Deux sources : `photo_format_error` (format lu par Pillow dans l'en-tête, comparé à `FORMAT_EXTENSIONS`, sinon à `Image.registered_extensions()` ; MPO accepte `.jpg`), appelée par `scan_worker` pour les photos ayant des visages (non mémorisée en base, refaite à chaque analyse), et les lignes `Error: … - <photo>` d'ExifTool pendant l'écriture. `write_worker` revérifie le format, **écarte** les photos mal nommées (ExifTool refuse « Not a valid JPG (looks more like a PNG) ») et envoie l'événement `errors` (`None` efface l'erreur, par exemple après une écriture vérifiée). Une photo en erreur reste « à écrire » dans la colonne « Métadonnées ». Non vérifié avec un vrai ExifTool ni dans une vraie fenêtre (testé avec un faux ExifTool et un Tkinter factice).
- **Emojis de couleur** : Tk dessine les emojis en noir et blanc (Windows), 🟩 🟦 🟥 ne distinguent donc rien. Leur couleur (`EMOJI_TEXT_COLORS`, mêmes teintes que `FACE_FRAME_COLORS`) passe sur le texte : une ligne qui commence par l'un d'eux est colorée en entier (`line_color`, infobulles et journal, ce dernier avec une étiquette `tk.Text` par couleur) ; dans le titre de la personne, seul le morceau qui suit l'emoji, jusqu'à la virgule ou la parenthèse, l'est (`colored_segments`). Le titre est un `ColoredLabel` (un `ttk.Label` par morceau, à mettre à jour par `set_text`), dont l'infobulle est attachée à chaque morceau. La couleur est lue **avant** `emoji()` : avec `--no-emoji`, elle reste. Les infobulles ont désormais une étiquette par ligne.
- **Photos choisies** : « Photos… » remplit `chosen_photos` et affiche `chosen_label` dans le champ « Dossier » ; tant que le champ affiche ce libellé, « Analyser » traite ces photos (`using_chosen_photos`). Les chemins passent par `os.path.abspath`, comme ceux de `list_photos`, pour retrouver les mêmes clés en base.
- **Thème sombre** : avec `--theme auto`, il est choisi d'après la valeur `AppsUseLightTheme` du registre ; sous macOS, `auto` garde le thème natif, qui suit déjà le système. `apply_dark_theme()` passe ttk sur le thème `clam` (les thèmes natifs `vista` et `winnative` ne se recolorent pas) et remplit la base d'options pour les widgets Tk classiques : elle doit donc être appelée **avant** la création des widgets. Le thème clair garde le thème natif. Les couleurs sont dans `DARK_COLORS`.
	- Barre de titre : appeler `set_title_bar_theme()` (attribut DWM 20) pour **chaque** fenêtre, y compris toute nouvelle boîte de dialogue.
	- Le journal est un `tk.Text` avec un `ttk.Scrollbar`, car l'ascenseur natif de `ScrolledText` ne se recolore pas.
	- Restent clairs : les `messagebox` et le sélecteur de dossier, qui sont natifs. Le thème est lu au démarrage, sans bascule en cours d'exécution.
- **Écrans à haute résolution** : `enable_dpi_awareness()` (mode « système ») est appelée avant la création de la fenêtre. Tk agrandit alors seul les polices (exprimées en points), mais toute taille en **pixels** doit passer par `scaled()` (`SETTINGS["scale"]`, qui vaut 1.25 à 125 %).
	- Les vignettes restent stockées en 96 px et sont agrandies à l'affichage par `thumbnail_photo()`.
	- La fenêtre est limitée à la zone utile de l'écran (`work_area()`, barre des tâches exclue) et centrée dedans.
	- Poste de Lionel : 1920×1080 à 125 %, Windows en mode sombre, couleur d'accentuation affichée sur les barres de titre (elle l'emporte sur la barre sombre quand la fenêtre est active).
- **Infobulles** : `ToolTip(widget, texte)` (Tkinter n'en a pas). Le texte est fixe, ou une fonction qui reçoit l'événement et renvoie le texte ou `None` (`tree_tooltip` : rôle de la colonne sur les en-têtes, résumé de la personne sur les lignes). Toute nouvelle commande, case ou réglage de l'interface doit recevoir la sienne, avec des emojis en tête de ligne.
	- Apparition après 500 ms d'immobilité (`TOOLTIP_DELAY`), disparition au clic jusqu'à la sortie de la souris. Fenêtre sans cadre, au premier plan, maintenue dans `work_area()`. Couleurs dans `TOOLTIP_COLORS`.
	- Les liaisons sont ajoutées avec `add = "+"` : créer l'infobulle **après** les autres `bind` du widget (un `bind` sans `add` remplacerait le sien). Les clics sont liés bouton par bouton : Tk n'exécute, pour un widget, que la liaison la plus précise, et un `<ButtonPress>` générique serait ignoré au profit d'un `<Button-3>` existant.
	- Le texte passe par `emoji()` ligne par ligne, et U+FE0F en est retiré : Tk dessine ce sélecteur de variante comme un blanc d'environ 21 px après 🏷️, 🖼️, 🖱️, ✏️, ⬆️… (mesuré). Les textes des autres widgets ont encore ce blanc (« 🏷️ étiquette » dans la liste, « 🎚️ Seuil de similarité »).
	- `EMOJI_PATTERN` ne couvre pas tous les emojis : ℹ️ (U+2139), ◀️ et ▶️ (U+25C0, U+25B6) restent avec `--no-emoji`. Vérifier qu'un nouvel emoji est dans ses plages.

### Modèles et DeepFace

- DeepFace est importé **à la première analyse** (`load_deepface`), pas au démarrage : son import (onnxruntime, OpenCV) prend plusieurs secondes.
- `load_deepface` fixe `DEEPFACE_HOME` (dossier de données) et `DEEPFACE_BACKEND_ENGINE` (`onnx`) **avant** l'import, sauf si l'utilisateur les a définis. Une valeur **vide** compte comme absente : avec `DEEPFACE_HOME=""`, DeepFace créait un `.deepface` relatif au dossier courant (constaté).
- Sans `DEEPFACE_BACKEND_ENGINE`, DeepFace préférerait TensorFlow puis PyTorch s'ils sont installés : les empreintes seraient légèrement différentes.
- L'image est passée à `represent` en tableau **BGR** déjà redressé (`exif_transpose`) : le chemin passerait par `cv2.imread`, qui échoue sur les chemins accentués sous Windows. `enforce_detection = False` : sans visage, DeepFace renvoie l'image entière avec une confiance nulle, écartée par `confidence <= 0`.
- Le logger de DeepFace écrit des emojis sur `sys.stdout` : une console en cp1252 lève `UnicodeEncodeError` (constaté dans un programme de test). `main()` passe la console en UTF-8.

## Tests

**Il n'y a pas encore de tests automatisés dans le dépôt.** Lors du développement (sous Linux, avec un faux moteur de détection) ont été vérifiés : regroupement, attribution des noms, écriture et relecture des métadonnées (accents inclus), idempotence de l'écriture, base SQLite, restauration, pagination, flux complet de l'interface, build PyInstaller.

Passage à DeepFace (novembre 2026), vérifié sous Windows avec les **vrais modèles** sur les images d'exemple publiques de DeepFace (copies, dossier temporaire) : téléchargement des poids, détection, regroupement, analyse complète par l'interface, sélection simple / Ctrl / Maj / tout / aucune, déplacement, détachement, mise à l'écart, validation d'une personne (seules ses photos sont écrites, `*_original` créés, seconde validation « déjà à jour »), analyse de photos choisies, capture de l'interface en thème sombre, programme compilé par PyInstaller. Non vérifiés : le thème clair et `--no-emoji` avec les nouveaux widgets, l'exécutable `--onefile --noconsole` complet, la CI (Linux, macOS, Python 3.11).

Pour en ajouter (pytest, dossier `tests/`) :

- Fonctions pures : `cluster_faces` (empreintes synthétiques), `parse_photo_tags`, `build_region_info`, `check_written`, `iou`.
- `Database` et `restore_originals` : sur des **copies** de photos, avec ExifTool.
- Interface : remplacer `face_tagger.FaceEngine` et `face_tagger.load_deepface` par un faux moteur renvoyant des empreintes synthétiques (octets `float32`) et des vignettes JPEG ; remplacer `messagebox.askyesno` et `NameDialog` ; piloter `start_scan`, `click_face`, `move_selected_faces`, `validate_person`, `start_write`, `start_restore` ; sous Linux, utiliser `xvfb-run`.
- Images de test publiques : `https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img1.jpg` (et `img2` à `img10`, `couple.jpg`).
- **Ne jamais tester sur de vraies photos** : toujours des copies, en gardant les `*_original`.
- Vérification visuelle sous Windows (méthode déjà employée pour le thème sombre et la haute résolution) :
	- lancer avec un `--data-directory` temporaire ;
	- remplir `app.library` de personnes fictives et remplacer `app.database.thumbnails` par des vignettes générées ;
	- capturer la fenêtre avec `PIL.ImageGrab`, en obtenant son rectangle par `DwmGetWindowAttribute(…, 9, …)`.
	- Ces fenêtres s'ouvrent sur l'écran de Lionel : limiter le nombre de lancements.
	- Infobulles : les déclencher par `event_generate("<Enter>")` puis `("<Motion>")` avec `x`, `y`, `rootx`, `rooty`, sans déplacer le vrai pointeur. Placer la fenêtre **loin du vrai pointeur** (`GetCursorPos`) : sinon ses événements réels se mêlent aux simulés. Écran verrouillé : `ImageGrab` capture l'écran de verrouillage.

## Points à traiter

- Ajouter des tests et compléter le `README.md` (installation, usage, captures).
- Supprimer le fichier obsolète `~/.face_tagger/faces_cache.pkl` (ancien cache de la v1) et, désormais, le dossier `~/.face_tagger/models` (modèles YuNet et SFace de la version OpenCV).
- Afficher dans une `messagebox` les erreurs de démarrage en mode fenêtré.
- Retirer U+FE0F des textes des widgets (blanc après certains emojis, voir « Infobulles ») et compléter `EMOJI_PATTERN` (ℹ️ notamment, déjà employé dans les messages).
- Vérifier la CI après le passage à DeepFace (taille des exécutables, `onnxruntime` sous macOS).
- Tester les vrais modèles sur un petit dossier de photos (copies), avec la sauvegarde `*_original` activée.
