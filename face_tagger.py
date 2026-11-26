#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regroupe les visages d'une photothèque par personne et inscrit les noms dans les métadonnées des photos.

Module face_tagger.py, application principale de Face Tagger :

	1. Parcours récursif d'un dossier de photos, ou analyse de quelques photos choisies une à une.
	2. Détection des visages et calcul d'une empreinte par visage avec DeepFace : détecteur RetinaFace et modèle de reconnaissance ArcFace par défaut (--detector, --model), exécutés par onnxruntime (ni TensorFlow, ni GPU requis).
	3. Regroupement des visages par personne (similarité cosinus).
	4. Les étiquettes déjà présentes dans les photos (régions de visages MWG / Microsoft, PersonInImage, éventuellement mots-clés) servent à nommer automatiquement les groupes.
	5. Les personnes non reconnues peuvent être nommées à la main dans une boîte de dialogue.
	6. Les vignettes d'une personne se sélectionnent à plusieurs (Ctrl / Maj + clic) pour être déplacées, détachées ou écartées d'un coup.
	7. Les noms sont validés personne par personne : ExifTool écrit XMP-iptcExt:PersonInImage, les régions de visages XMP-mwg-rs, les mots-clés XMP-dc:Subject et EXIF XPKeywords (visible dans l'Explorateur Windows), puis relit chaque photo pour vérifier que les noms y sont.
	8. Une base SQLite (face_database.py) mémorise visages, vignettes, étiquettes lues et historique des modifications : une photo inchangée n'est jamais ré-analysée ni relue.
	9. Les visages d'une personne sont affichés page par page (seules les vignettes de la page courante sont chargées en mémoire).
	10. Les sauvegardes *_original d'ExifTool se restaurent d'un coup (bouton, ou restore_originals.py).
	11. L'interface suit le mode sombre de Windows (ou --theme) et reste nette sur les écrans à haute résolution.
	12. Une infobulle explique chaque commande, réglage et colonne de l'interface.

Fichiers à placer dans le même dossier : face_tagger.py, face_database.py, restore_originals.py

Dépendances :
	pip install --requirement requirements.txt   (deepface, pillow, numpy, pillow-heif pour les HEIC)
	ExifTool   (Fedora : sudo dnf install perl-Image-ExifTool)
	Tkinter    (Fedora : sudo dnf install python3-tkinter)

Example:
	python face_tagger.py --directory ~/Images --model ArcFace --detector retinaface
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import ModuleType
from typing import Callable, Iterable, Optional

if sys.platform == "win32":
	# Lecture du thème de Windows (clair ou sombre)
	import winreg

# DeepFace n'est pas importé ici mais à la première analyse (voir load_deepface) : son import charge onnxruntime et OpenCV, ce qui retarderait de plusieurs secondes l'ouverture de la fenêtre, même pour une photothèque déjà analysée qui n'en a pas besoin.
try:
	import numpy as np
	from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk
except ImportError as error:
	sys.exit(f"❌ Dépendance manquante : {error}\n👉 Installez-la avec : pip install --requirement requirements.txt")

# Prise en charge optionnelle des HEIC/HEIF (photos d'iPhone) si pillow-heif est installé
try:
	import pillow_heif
	pillow_heif.register_heif_opener()
	HEIF_AVAILABLE: bool = True
except ImportError:
	HEIF_AVAILABLE = False

# Modules compagnons (même dossier) : base SQLite et restauration des sauvegardes
try:
	from face_database import Database
	import restore_originals
except ImportError as error:
	sys.exit(f"❌ Fichier manquant : {error}\n👉 Placez face_tagger.py, face_database.py et restore_originals.py dans le même dossier.")


# ======================================================================================
# Constantes
# ======================================================================================

# Extensions d'images analysées
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
if HEIF_AVAILABLE:
	IMAGE_EXTENSIONS |= {".heic", ".heif"}

# Extensions admises pour chaque format d'image reconnu par Pillow (d'après le contenu du fichier, voir photo_format_error).
# Les autres formats se rabattent sur les extensions que Pillow leur associe (Image.registered_extensions).
FORMAT_EXTENSIONS: dict[str, tuple[str, ...]] = {
	"JPEG": (".jpg", ".jpeg", ".jpe", ".jfif"),
	"MPO": (".jpg", ".jpeg", ".mpo"),      # JPEG à plusieurs images (appareils photo 3D, certains smartphones)
	"PNG": (".png",),
	"TIFF": (".tif", ".tiff"),
	"WEBP": (".webp",),
	"HEIF": (".heic", ".heif"),             # Format annoncé par pillow-heif
}

# Modèles de reconnaissance de DeepFace proposés, avec leur seuil de similarité cosinus par défaut. Chaque modèle
# a sa propre échelle de ressemblance : ces seuils sont ceux de DeepFace (deepface/config/threshold.py, exprimés
# en distance cosinus), convertis en similarité (1 − distance). Tous s'exécutent avec onnxruntime.
RECOGNITION_MODELS: dict[str, float] = {
	"ArcFace": 0.32,            # 512 dimensions : le meilleur compromis précision / vitesse (choix par défaut)
	"Facenet512": 0.70,         # 512 dimensions : très précis, seuil élevé
	"GhostFaceNet": 0.35,       # 512 dimensions : léger
	"SFace": 0.41,              # 128 dimensions : le modèle de l'ancienne version (OpenCV)
	"Facenet": 0.60,            # 128 dimensions
	"VGG-Face": 0.32,           # 4096 dimensions : lent et volumineux
}
DEFAULT_MODEL: str = "ArcFace"

# Détecteurs de visages de DeepFace proposés. RetinaFace repère bien les visages petits, de profil ou dans la
# pénombre, mais il est lent : environ 3,5 s par photo sur le processeur, contre 0,35 s pour YuNet (mesuré sur
# des photos de 2000 pixels, mêmes visages trouvés).
DETECTOR_BACKENDS: tuple[str, ...] = ("retinaface", "yunet", "centerface", "mtcnn", "ssd", "opencv")
DEFAULT_DETECTOR: str = "retinaface"

# Moteur d'exécution des réseaux de neurones imposé à DeepFace (s'il n'est pas déjà fixé par la variable
# d'environnement) : onnxruntime est installé avec DeepFace, alors que TensorFlow ou PyTorch, s'ils sont présents
# sur le poste, seraient choisis en priorité et donneraient des empreintes légèrement différentes.
DEEPFACE_ENGINE: str = "onnx"

# Bornes du curseur de similarité : assez larges pour couvrir les seuils de tous les modèles ci-dessus
THRESHOLD_MIN: float = 0.20
THRESHOLD_MAX: float = 0.80

THUMBNAIL_SIZE: int = 96          # Côté (en pixels) des vignettes de visages
FACE_FRAME_WIDTH: int = 3         # Épaisseur (en pixels, écran à 100 %) du cadre coloré des vignettes de visages
TREE_ROW_HEIGHT: int = 44         # Hauteur des lignes de la liste des personnes
ICON_CACHE_SIZE: int = 4000       # Nombre maximal d'icônes de la liste des personnes gardées en mémoire
TOOLTIP_DELAY: int = 500          # Délai (en millisecondes) avant l'apparition d'une infobulle, celui de Windows
TOOLTIP_WIDTH: int = 380          # Largeur maximale (en pixels, écran à 100 %) du texte d'une infobulle
REGION_MATCH_IOU: float = 0.15      # Recouvrement minimal entre un visage détecté et une région existante
EXIFTOOL_BATCH: int = 25          # Nombre de photos modifiées par appel à ExifTool

# Polices d'emojis en couleur essayées pour dessiner un emoji dans une image (voir emoji_image), avec la taille de rendu.
# Les polices à images intégrées n'existent qu'en quelques tailles fixes : toute autre taille fait échouer leur chargement.
EMOJI_FONTS: tuple[tuple[str, int], ...] = (
	("seguiemj.ttf", 128),              # Windows : Segoe UI Emoji, vectorielle (toute taille convient)
	("Apple Color Emoji.ttc", 160),     # macOS : images intégrées, 160 est l'une de leurs tailles
	("NotoColorEmoji.ttf", 109),        # Linux : images intégrées, en une seule taille
)

# Identifiant de l'application pour la barre des tâches de Windows (voir set_windows_app_id)
APP_USER_MODEL_ID: str = "ILPlais.FaceTagger"

# Couleurs du thème sombre (proches de celles de Windows 11)
DARK_COLORS: dict[str, str] = {
	"background": "#202020",        # Fond des fenêtres
	"surface": "#2b2b2b",           # Champs de saisie, listes et journal
	"button": "#333333",            # Boutons, en-têtes de colonnes, ascenseurs
	"hover": "#3e3e3e",             # Élément survolé
	"pressed": "#4a4a4a",           # Bouton enfoncé
	"thumb": "#6b6b6b",             # Curseur du seuil et ascenseurs, plus clairs pour se détacher de leur glissière
	"thumb_hover": "#858585",
	"thumb_pressed": "#9e9e9e",
	"border": "#4a4a4a",            # Bordures
	"text": "#f0f0f0",              # Texte
	"disabled": "#7a7a7a",          # Texte des éléments désactivés
	"selection": "#0067c0",         # Fond de la sélection (bleu de Windows)
	"selection_text": "#ffffff",    # Texte sélectionné
	"accent": "#4cc2ff",            # Barre de progression
}

# Couleurs des infobulles pour chaque thème (le thème sombre reprend les teintes de DARK_COLORS)
TOOLTIP_COLORS: dict[str, dict[str, str]] = {
	"light": {
		"background": "#ffffff",
		"text": "#1a1a1a",
		"border": "#767676"
	},
	"dark": {
		"background": DARK_COLORS["surface"],
		"text": DARK_COLORS["text"],
		"border": DARK_COLORS["thumb"]
	},
}

# Mise en évidence des vignettes sélectionnées, pour chaque thème (bleu clair de l'Explorateur, bleu de Windows)
SELECTION_COLORS: dict[str, dict[str, str]] = {
	"light": {
		"background": "#cce4f7",
		"text": "#000000"
	},
	"dark": {
		"background": DARK_COLORS["selection"],
		"text": DARK_COLORS["selection_text"]
	},
}

# Cadre des vignettes de visages, identique dans les deux thèmes (teintes assez soutenues pour un fond clair comme sombre) :
# vert pour un visage déjà marqué dans la photo, bleu pour un visage seulement proposé (nom pas encore écrit),
# rouge pour un visage d'une photo en erreur (extension qui ne correspond pas au contenu, écriture refusée…)
FACE_FRAME_COLORS: dict[str, str] = {
	"tagged": "#2ea043",
	"proposed": "#2f81f7",
	"error": "#e5534b",
}

# Couleur du texte introduit par un emoji de couleur. Tk dessine les emojis en noir et blanc (sous Windows) :
# 🟩 🟦 🟥 n'y distinguent plus rien, leur couleur est donc reportée sur le texte qu'ils introduisent (titre de la
# personne, infobulles, journal). Mêmes teintes que le cadre des vignettes, lisibles sur fond clair comme sombre.
EMOJI_TEXT_COLORS: dict[str, str] = {
	"🟩": FACE_FRAME_COLORS["tagged"],
	"🟦": FACE_FRAME_COLORS["proposed"],
	"🟥": FACE_FRAME_COLORS["error"],
}

# Opacité (de 0, invisible, à 1, opaque) du badge 💾 des visages déjà enregistrés, dans l'aperçu d'une photo (voir saved_badge) :
# assez transparent pour laisser voir le visage en dessous, le fond l'étant plus que la disquette pour qu'elle reste lisible
SAVED_BADGE_OPACITY: dict[str, float] = {
	"background": 0.35,
	"icon": 0.50,
}

# Réglages globaux : emojis et thème viennent de la ligne de commande, l'échelle de l'écran est mesurée au démarrage
SETTINGS: dict[str, bool | float] = {"emoji": True, "dark": False, "scale": 1.0}

# Caractères considérés comme des emojis (utile pour l'option --no-emoji)
EMOJI_PATTERN: re.Pattern[str] = re.compile("[\U0001F000-\U0001FFFF\u2190-\u21FF\u23E9-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")

# Étiquettes ExifTool lues dans chaque photo
TAGS_TO_READ: list[str] = [
	"-XMP-mwg-rs:RegionInfo",        # Régions de visages (norme MWG : Lightroom, digiKam, Picasa…)
	"-XMP-MP:RegionInfoMP",          # Régions de visages « Microsoft Photo »
	"-XMP-iptcExt:PersonInImage",    # Liste de personnes présentes (sans position)
	"-XMP-dc:Subject",               # Mots-clés XMP
	"-IPTC:Keywords",                # Mots-clés IPTC
	"-XPKeywords",                   # Mots-clés EXIF de Windows
]


# ======================================================================================
# Fonctions utilitaires
# ======================================================================================

def emoji(text: str) -> str:
	"""Adapte un texte destiné à l'utilisateur à l'option --no-emoji.

	Args:
		text (str): Texte à afficher, avec ses emojis.

	Returns:
		str: Le texte tel quel ou, avec --no-emoji, débarrassé de ses emojis et des blancs à ses extrémités.
	"""
	if SETTINGS["emoji"]:
		return text
	return EMOJI_PATTERN.sub("", text).strip()

def resource_path(relative_path: str) -> str:
	"""Donne le chemin absolu d'une ressource livrée avec l'application (icône…).

	Sous PyInstaller, les ressources sont extraites dans un dossier temporaire (`sys._MEIPASS`). En développement, elles sont cherchées dans le dossier du script, et non dans le dossier courant, pour être trouvées quel que soit le dossier de lancement.

	Args:
		relative_path (str): Chemin de la ressource, relatif au dossier de l'application.

	Returns:
		str: Chemin absolu de la ressource (son existence n'est pas vérifiée).
	"""
	try:
		# PyInstaller crée un dossier temporaire et stocke le chemin dans _MEIPASS
		base_path = sys._MEIPASS
	except AttributeError:
		# En développement : dossier du script, et non le dossier courant, pour trouver l'icône même quand l'application est lancée depuis un autre dossier (raccourci, autre terminal…)
		base_path = os.path.dirname(os.path.abspath(__file__))

	return os.path.join(base_path, relative_path)


def set_windows_app_id() -> None:
	"""Donne à l'application sa propre identité dans la barre des tâches de Windows.

	Lancé par « `python face_tagger.py` », le programme est vu par Windows comme `python.exe` : la barre des tâches regroupe alors sa fenêtre avec Python et affiche l'icône de Python au lieu de celle de la fenêtre. Un identifiant d'application explicite (`AppUserModelID`) l'en sépare. L'identifiant vaut pour tout le processus et doit être déclaré avant la création de la première fenêtre.
	"""
	if sys.platform != "win32":
		return
	try:
		ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
	except (AttributeError, OSError):
		# Fonction absente ou appel refusé : seule l'icône de la barre des tâches est en jeu, on continue
		pass


def enable_dpi_awareness() -> None:
	"""Déclare à Windows que l'application gère elle-même les écrans à haute résolution (125 %, 150 %…).

	Sans cette déclaration, Windows dessine la fenêtre en 96 ppp puis l'agrandit comme une image : textes et vignettes sont flous. Une fois déclarée, Tk mesure la vraie résolution et agrandit seul les polices ; les tailles exprimées en pixels sont agrandies par scaled(). Le mode « système » est préféré au mode « par écran » : Tk ne sait pas se redimensionner quand la fenêtre passe sur un écran d'une autre résolution, c'est alors Windows qui s'en charge. Doit être appelée avant la création de la première fenêtre.
	"""
	if sys.platform != "win32":
		return
	try:
		# PROCESS_SYSTEM_DPI_AWARE (Windows 8.1 et suivants). Un refus (déjà déclaré par le manifeste de l'exécutable) est renvoyé comme code d'erreur, sans exception : il est sans conséquence.
		ctypes.windll.shcore.SetProcessDpiAwareness(1)
	except (AttributeError, OSError):
		try:
			# Windows Vista à 8
			ctypes.windll.user32.SetProcessDPIAware()
		except (AttributeError, OSError):
			pass


def windows_prefers_dark() -> bool:
	"""Indique si Windows est réglé en mode sombre pour les applications.

	Le réglage (Paramètres > Personnalisation > Couleurs) est lu dans la valeur `AppsUseLightTheme` du registre.

	Returns:
		bool: `True` en mode sombre ; False en mode clair, hors de Windows, ou si la valeur est absente (Windows antérieur à 10).
	"""
	if sys.platform != "win32":
		return False
	try:
		with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
			return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
	except OSError:
		# Valeur absente (Windows antérieur à 10) : thème clair
		return False


def strip_widget_emoji(widget: tk.Misc) -> None:
	"""Retire récursivement les emojis des textes d'un widget et de ses enfants (option `--no-emoji`).

	Sans `--no-emoji`, la fonction ne fait rien.

	Args:
		widget (tk.Misc): Widget (ou fenêtre) à traiter, enfants compris.
	"""
	if SETTINGS["emoji"]:
		return
	for child in widget.winfo_children():
		strip_widget_emoji(child)
	try:
		text = str(widget.cget("text"))
	except tk.TclError:
		return  # Ce widget n'a pas d'option « text »
	if text:
		widget.configure(text = emoji(text))


def normalize_name(name: str) -> str:
	"""Supprime les espaces superflus d'un nom : « Jean␣␣␣Dupont » devient « Jean␣Dupont ».

	Args:
		name (str): Nom à nettoyer (une autre valeur est d'abord convertie en texte).

	Returns:
		str: Nom sans espace en tête, en fin ni en double.
	"""
	return " ".join(str(name).split())


def name_key(name: str) -> str:
	"""Calcule la clé de comparaison d'un nom, insensible à la casse et aux espaces superflus.

	Args:
		name (str): Nom à comparer.

	Returns:
		str: Clé de comparaison : « jean␣dupont » pour « Jean␣␣DUPONT ».
	"""
	return normalize_name(name).casefold()


def merge_unique(*lists: Iterable[str]) -> list[str]:
	"""Fusionne des listes de noms sans doublon, en gardant l'ordre de première apparition.

	Les doublons sont repérés sans tenir compte de la casse ni des espaces superflus ; les noms vides sont écartés.

	Args:
		*lists (Iterable[str]): Listes de noms à fusionner, dans l'ordre.

	Returns:
		list[str]: Noms normalisés (voir `normalize_name`), chacun présent une seule fois.
	"""
	seen: set[str] = set()
	result: list[str] = []
	for items in lists:
		for item in items:
			item = normalize_name(item)
			if item and name_key(item) not in seen:
				seen.add(name_key(item))
				result.append(item)
	return result


def as_list(value: object) -> list[str]:
	"""Convertit une valeur renvoyée par ExifTool en liste de textes.

	ExifTool renvoie une valeur seule (texte ou nombre) quand un champ de type liste n'a qu'un élément.

	Args:
		value (object): Valeur lue : `None`, texte, nombre ou liste.

	Returns:
		list[str]: Liste vide pour `None`, liste des éléments convertis en textes, ou liste d'un seul élément.
	"""
	if value is None:
		return []
	if isinstance(value, list):
		return [str(item) for item in value]
	return [str(value)]


def path_key(path: str) -> str:
	"""Calcule la clé normalisée d'un chemin, pour faire correspondre les chemins de Python et ceux d'ExifTool.

	Args:
		path (str): Chemin d'un fichier, quels que soient ses séparateurs.

	Returns:
		str: Chemin normalisé ; sous Windows, séparateurs et casse sont unifiés.
	"""
	return os.path.normcase(os.path.normpath(path))


def iou(box_a: tuple, box_b: tuple) -> float:
	"""Calcule l'intersection sur union (IoU) de deux rectangles.

	Args:
		box_a (tuple): Premier rectangle (x, y, largeur, hauteur), (x, y) étant le coin supérieur gauche.
		box_b (tuple): Second rectangle, dans le même repère.

	Returns:
		float: Aire commune divisée par l'aire réunie des deux rectangles, de `0.0` (disjoints) à `1.0` (identiques).
	"""
	ax, ay, aw, ah = box_a
	bx, by, bw, bh = box_b
	inter_w = min(ax + aw, bx + bw) - max(ax, bx)
	inter_h = min(ay + ah, by + bh) - max(ay, by)
	if inter_w <= 0 or inter_h <= 0:
		return 0.0
	intersection = inter_w * inter_h
	return intersection / (aw * ah + bw * bh - intersection)


def list_photos(directory: str) -> list[str]:
	"""Liste récursivement les photos d'un dossier.

	Les dossiers et fichiers cachés (nom commençant par un point) sont ignorés, et seules les extensions de `IMAGE_EXTENSIONS` sont retenues.

	Args:
		directory (str): Dossier à parcourir, sous-dossiers compris.

	Returns:
		list[str]: Chemins des photos, triés par dossier puis par nom.
	"""
	photos: list[str] = []
	for root, directories, files in os.walk(directory):
		# Modifier la liste « directories » sur place permet d'élaguer le parcours
		directories[:] = sorted(d for d in directories if not d.startswith("."))
		for file_name in sorted(files):
			if file_name.startswith("."):
				continue
			if Path(file_name).suffix.lower() in IMAGE_EXTENSIONS:
				photos.append(os.path.join(root, file_name))
	return photos


def photo_format_error(photo: str) -> Optional[str]:
	"""Vérifie que l'extension d'une photo correspond à son contenu (une image PNG nommée `*.jpg`, par exemple).

	Pillow reconnaît le format d'après les premiers octets du fichier, sans tenir compte de son nom : une photo mal nommée s'analyse donc normalement. ExifTool, lui, refuse d'écrire dans un fichier dont l'extension ne correspond pas au contenu (« Not a valid JPG (looks more like a PNG) ») : mieux vaut le signaler avant l'écriture. Seul l'en-tête est lu : la vérification est rapide.

	Args:
		photo (str): Chemin de la photo.

	Returns:
		Optional[str]: Description de l'erreur, ou `None` si l'extension convient (ou si le format n'est pas reconnu par Pillow : l'analyse l'aura déjà signalé).
	"""
	try:
		with Image.open(photo) as image:
			image_format = (image.format or "").upper()
	except Exception:
		return None
	if not image_format:
		return None
	extension = Path(photo).suffix.lower()
	expected = FORMAT_EXTENSIONS.get(image_format) or tuple(sorted(
		suffix for suffix, name in Image.registered_extensions().items() if name.upper() == image_format
	))
	if not expected or extension in expected:
		return None
	return (
		f"Extension « {extension or '(aucune)'} » incorrecte : ce fichier est en réalité une image {image_format}. "
		f"Renommez-le en « {Path(photo).stem}{expected[0]} » : ExifTool refuse d'écrire les noms dans un fichier "
		"dont l'extension ne correspond pas au contenu."
	)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
	"""Charge une police lisible pour annoter les photos.

	DejaVu Sans puis Arial sont essayées ; la police par défaut de Pillow sert en dernier recours.

	Args:
		size (int): Taille de la police, en pixels.

	Returns:
		ImageFont.FreeTypeFont | ImageFont.ImageFont: Police TrueType trouvée, ou police par défaut.
	"""
	for font_name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
		try:
			return ImageFont.truetype(font_name, size)
		except OSError:
			continue
	return ImageFont.load_default()


@functools.lru_cache(maxsize = 8)
def emoji_image(character: str) -> Optional[Image.Image]:
	"""Dessine un emoji en couleur avec la première police d'emojis du système trouvée (voir `EMOJI_FONTS`).

	Sert à poser un emoji sur une photo : Pillow le dessine avec les couleurs de la police (`embedded_color`). Le résultat est gardé en cache : chaque emoji n'est dessiné qu'une fois, puis seulement réduit.

	Args:
		character (str): Emoji à dessiner (« 💾 »…).

	Returns:
		Optional[Image.Image]: Emoji en RGBA sur fond transparent, carré, à la taille de rendu de la police (une centaine de pixels) ; `None` si aucune police d'emojis n'a pu être chargée.
	"""
	for font_name, size in EMOJI_FONTS:
		try:
			font = ImageFont.truetype(font_name, size)
		except OSError:
			# Police absente de ce système, ou taille non disponible
			continue
		# Toile deux fois plus grande que la police : certains emojis débordent de leur boîte nominale
		canvas = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
		try:
			ImageDraw.Draw(canvas).text((size // 2, size // 2), character, font = font, embedded_color = True)
		except (OSError, ValueError):
			continue
		box = canvas.getbbox()
		if box is None:
			continue
		# Recadré sur le dessin puis centré dans un carré, pour être réduit sans déformation
		glyph = canvas.crop(box)
		side = max(glyph.size)
		square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
		square.paste(glyph, ((side - glyph.width) // 2, (side - glyph.height) // 2))
		return square
	return None


def floppy_pictogram(size: int) -> Image.Image:
	"""Dessine une disquette simplifiée, qui remplace l'emoji 💾 sans police d'emojis ou avec l'option `--no-emoji`.

	Le dessin est fait quatre fois plus grand puis réduit : ses bords sont ainsi lissés.

	Args:
		size (int): Côté de l'icône, en pixels.

	Returns:
		Image.Image: Disquette en RGBA sur fond transparent.
	"""
	big = size * 4
	image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
	draw = ImageDraw.Draw(image)
	margin = big // 16
	# Corps de la disquette, volet métallique (avec sa fenêtre) en haut, étiquette blanche en bas
	draw.rounded_rectangle([margin, margin, big - margin, big - margin], radius = big // 10, fill = "#3a4a5c")
	draw.rectangle([big * 0.28, margin, big * 0.72, big * 0.38], fill = "#d0d4d8")
	draw.rectangle([big * 0.56, big * 0.12, big * 0.66, big * 0.32], fill = "#3a4a5c")
	draw.rectangle([big * 0.2, big * 0.52, big * 0.8, big - margin * 2], fill = "#ffffff")
	return image.resize((size, size), Image.Resampling.LANCZOS)


def saved_icon(size: int) -> Image.Image:
	"""Fournit l'icône 💾 qui marque, dans l'aperçu d'une photo, un visage déjà enregistré dans ses métadonnées.

	L'emoji est dessiné avec la police d'emojis du système ; sans elle, ou avec l'option `--no-emoji`, une disquette simplifiée le remplace.

	Args:
		size (int): Côté de l'icône, en pixels.

	Returns:
		Image.Image: Icône carrée en RGBA, à coller avec son canal alpha comme masque.
	"""
	icon = emoji_image("💾") if SETTINGS["emoji"] else None
	if icon is None:
		return floppy_pictogram(size)
	return icon.resize((size, size), Image.Resampling.LANCZOS)


def saved_badge(size: int, padding: int) -> Image.Image:
	"""Compose le badge translucide 💾 posé sur les visages déjà enregistrés, dans l'aperçu d'une photo.

	Le badge (fond blanc arrondi et disquette) est rendu translucide selon `SAVED_BADGE_OPACITY` : le visage reste visible en dessous. Le fond est plus transparent que la disquette, pour la détacher d'une photo sombre sans masquer le visage.

	Args:
		size (int): Côté de la disquette, en pixels.
		padding (int): Marge entre la disquette et le bord du fond, en pixels.

	Returns:
		Image.Image: Badge carré en RGBA, de côté `size + 2 * padding`, à coller avec son canal alpha comme masque.
	"""
	side = size + 2 * padding
	badge = Image.new("RGBA", (side, side), (0, 0, 0, 0))
	ImageDraw.Draw(badge).rounded_rectangle(
		[0, 0, side - 1, side - 1],
		radius = 2 * padding,
		fill = (255, 255, 255, round(255 * SAVED_BADGE_OPACITY["background"]))
	)
	icon = saved_icon(size)
	# L'opacité de la disquette s'applique à son propre canal alpha : ses bords lissés restent progressifs
	icon.putalpha(icon.getchannel("A").point(lambda alpha: round(alpha * SAVED_BADGE_OPACITY["icon"])))
	badge.alpha_composite(icon, (padding, padding))
	return badge


def make_thumbnail(
		image: Image.Image,
		box: tuple[float, float, float, float]
	) -> Image.Image:
	"""Découpe une vignette carrée autour d'un visage, avec une marge, pour l'affichage.

	Args:
		image (Image.Image): Photo redressée, à la taille utilisée pour la détection.
		box (tuple[float, float, float, float]): Rectangle du visage, en pixels : (gauche, haut, droite, bas).

	Returns:
		Image.Image: Vignette carrée de `THUMBNAIL_SIZE` pixels de côté.
	"""
	x0, y0, x1, y1 = box
	center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
	half = max(x1 - x0, y1 - y0) * 0.75
	crop = image.crop((int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)))
	return ImageOps.fit(crop, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)


def encode_thumbnail(image: Image.Image) -> bytes:
	"""Encode une vignette en JPEG, pour la stocker dans la base SQLite.

	Args:
		image (Image.Image): Vignette à encoder.

	Returns:
		bytes: Contenu du fichier JPEG (qualité `85`).
	"""
	buffer = io.BytesIO()
	image.save(buffer, format = "JPEG", quality = 85)
	return buffer.getvalue()


def decode_thumbnail(data: Optional[bytes]) -> Image.Image:
	"""Décode une vignette JPEG lue dans la base.

	Args:
		data (Optional[bytes]): Contenu JPEG de la vignette, ou `None` (ou vide) si elle est absente.

	Returns:
		Image.Image: Vignette en RVB, ou carré gris de `THUMBNAIL_SIZE` pixels si elle est absente.
	"""
	if not data:
		return Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), "#888888")
	return Image.open(io.BytesIO(data)).convert("RGB")


def load_deepface(data_directory: Path) -> ModuleType:
	"""Importe DeepFace à la première analyse et lui indique où ranger ses modèles.

	DeepFace télécharge lui-même les poids de chaque réseau à sa première utilisation (quelques centaines de Mio), ici dans `<data_directory>/.deepface/weights` plutôt que dans le dossier personnel. Ses variables d'environnement doivent être fixées AVANT l'import, car il les lit à ce moment-là ; une valeur déjà définie par l'utilisateur est respectée. Sous Windows, le GPU est employé par DirectML si `onnxruntime-directml` est installé.

	Args:
		data_directory (Path): Dossier de données de l'application, où DeepFace crée son dossier « `.deepface` ».

	Returns:
		ModuleType: Le module `deepface.DeepFace`.

	Raises:
		RuntimeError: Si DeepFace, ou l'une de ses dépendances, ne peut pas être importé.
	"""
	# Une variable définie mais vide compte comme absente : DeepFace rangerait sinon ses modèles dans un dossier « .deepface » relatif au dossier courant
	if not os.environ.get("DEEPFACE_HOME", "").strip():
		os.environ["DEEPFACE_HOME"] = str(data_directory)
	if not os.environ.get("DEEPFACE_BACKEND_ENGINE", "").strip():
		os.environ["DEEPFACE_BACKEND_ENGINE"] = DEEPFACE_ENGINE
	# Sous Windows, DeepFace ne prend DirectML (GPU de n'importe quel fabricant via DirectX 12) que si on le lui demande xplicitement : par défaut il reste sur le processeur. On ne le demande que si onnxruntime-directml est installé,sinon DeepFace signalerait un avertissement inutile. Un réglage de l'utilisateur est toujours respecté.
	if sys.platform == "win32" and not os.environ.get("DEEPFACE_ONNX_PROVIDERS", "").strip():
		try:
			import onnxruntime
		except ImportError:
			# Pas d'onnxruntime du tout : l'erreur sera signalée à l'import de DeepFace, plus bas
			pass
		else:
			if "DmlExecutionProvider" in onnxruntime.get_available_providers():
				os.environ["DEEPFACE_ONNX_PROVIDERS"] = "directml,cpu"
	try:
		from deepface import DeepFace
	except ImportError as error:
		raise RuntimeError(
			f"DeepFace est introuvable ({error}).\n"
			"👉 Installez-le avec : pip install --requirement requirements.txt"
		) from error
	return DeepFace


# ======================================================================================
# ExifTool : lecture et écriture des métadonnées
# ======================================================================================

@dataclass
class PhotoTags:
	"""Étiquettes déjà présentes dans une photo, lues par ExifTool.

	Attributes:
		regions (list[dict]): Régions de visages MWG et Microsoft, chacune avec les clés « `name` » (`str`, parfois vide) et « `box` » (x, y, largeur, hauteur normalisés, (x, y) étant le coin supérieur gauche).
		region_info (Optional[dict]): Structure `XMP-mwg-rs:RegionInfo` brute, conservée pour la réécrire, ou `None`.
		person_in_image (list[str]): Noms de `XMP-iptcExt:PersonInImage` (sans position).
		subject (list[str]): Mots-clés `XMP-dc:Subject`.
		keywords (list[str]): Mots-clés `IPTC:Keywords`.
		xp_keywords (list[str]): Mots-clés EXIF `XPKeywords` de Windows.
		raw (dict): Entrée JSON brute d'ExifTool, sans `SourceFile` (mémorisée dans la base).
	"""
	regions: list[dict] = field(default_factory = list)          # [{"name", "box"}], box = (x, y, l, h) normalisé
	region_info: Optional[dict] = None                           # Structure MWG brute, conservée à l'écriture
	person_in_image: list[str] = field(default_factory = list)
	subject: list[str] = field(default_factory = list)
	keywords: list[str] = field(default_factory = list)
	xp_keywords: list[str] = field(default_factory = list)
	raw: dict = field(default_factory = dict)                    # Entrée JSON brute d'ExifTool (mémorisée en base)

	def weak_names(self, use_keywords: bool) -> list[str]:
		"""Liste les noms « candidats », présents dans la photo mais sans position dans l'image.

		Args:
			use_keywords (bool): `True` pour compter aussi les mots-clés (`Subject`, `Keywords`, `XPKeywords`) comme noms.

		Returns:
			list[str]: Noms de `PersonInImage`, et des mots-clés si demandé, sans doublon.
		"""
		names = list(self.person_in_image)
		if use_keywords:
			names += self.subject + self.keywords + self.xp_keywords
		return merge_unique(names)


def parse_photo_tags(entry: dict) -> PhotoTags:
	"""Transforme l'entrée JSON d'ExifTool pour une photo en objet PhotoTags.

	Les régions MWG (X et Y au centre) et Microsoft (coin supérieur gauche) sont converties dans le même repère : coin supérieur gauche, valeurs normalisées. Les régions illisibles ou qui ne sont pas des visages sont ignorées.

	Args:
		entry (dict): Entrée JSON d'ExifTool (options `-json -struct`) pour une photo ; un dictionnaire vide donne des étiquettes vides.

	Returns:
		PhotoTags: Étiquettes de la photo.
	"""
	tags = PhotoTags(raw = {key: value for key, value in entry.items() if key != "SourceFile"})

	# Régions MWG : X/Y désignent le CENTRE du rectangle, en coordonnées normalisées
	info = entry.get("RegionInfo")
	if isinstance(info, dict):
		tags.region_info = info
		regions = info.get("RegionList") or []
		if isinstance(regions, dict):
			regions = [regions]
		for region in regions:
			if not isinstance(region, dict) or str(region.get("Type", "Face")) != "Face":
				continue
			area = region.get("Area") or {}
			try:
				width, height = float(area["W"]), float(area["H"])
				center_x, center_y = float(area["X"]), float(area["Y"])
			except (KeyError, TypeError, ValueError):
				continue
			tags.regions.append({
				"name": normalize_name(region.get("Name", "")),
				"box": (center_x - width / 2, center_y - height / 2, width, height),
			})

	# Régions Microsoft : « Rectangle » = "x, y, largeur, hauteur" (coin supérieur gauche, normalisé)
	microsoft = entry.get("RegionInfoMP")
	if isinstance(microsoft, dict):
		regions = microsoft.get("Regions") or []
		if isinstance(regions, dict):
			regions = [regions]
		for region in regions:
			if not isinstance(region, dict):
				continue
			try:
				x, y, width, height = [float(v) for v in str(region.get("Rectangle", "")).replace(",", " ").split()]
			except ValueError:
				continue
			tags.regions.append({
				"name": normalize_name(region.get("PersonDisplayName", "")),
				"box": (x, y, width, height),
			})

	tags.person_in_image = as_list(entry.get("PersonInImage"))
	tags.subject = as_list(entry.get("Subject"))
	tags.keywords = as_list(entry.get("Keywords"))
	tags.xp_keywords = [part.strip() for part in str(entry.get("XPKeywords", "")).split(";") if part.strip()]
	return tags


class ExifTool:
	"""Petit adaptateur autour de l'exécutable ExifTool.

	Attributes:
		path (Optional[str]): Chemin complet de l'exécutable, ou None s'il est introuvable.
	"""

	def __init__(self, executable: str = "exiftool") -> None:
		"""Cherche l'exécutable d'ExifTool.

		Args:
			executable (str): Nom de l'exécutable (cherché dans le `PATH`) ou chemin complet.
		"""
		self.path: Optional[str] = shutil.which(executable)

	@property
	def available(self) -> bool:
		"""bool: True si l'exécutable d'ExifTool a été trouvé."""
		return self.path is not None

	def run(
			self,
			options: list[str],
			files: list[str],
			cwd: Optional[str] = None
		) -> subprocess.CompletedProcess:
		"""Lance ExifTool et attend la fin de son exécution.

		Les arguments (noms de fichiers compris) passent en UTF-8 par l'entrée standard (`-@ -`) : cela évite la limite de longueur de la ligne de commande et les soucis d'encodage des noms accentués, notamment sous Windows.

		Args:
			options (list[str]): Options d'ExifTool (par exemple `["-json", "-struct"]`).
			files (list[str]): Fichiers à traiter.
			cwd (Optional[str]): Dossier de travail du processus, ou `None` pour garder le dossier courant.

		Returns:
			subprocess.CompletedProcess: Résultat du processus, avec sa sortie standard et sa sortie d'erreur en octets.
		"""
		arguments = ["-charset", "filename=utf8", *options, *files]
		return subprocess.run(
			[self.path, "-@", "-"],
			input = "\n".join(arguments).encode("utf-8"),
			capture_output = True,
			cwd = cwd,
			# Sous Windows, évite le clignotement d'une fenêtre console à chaque appel (lancement via pythonw)
			creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
		)

	def read_tags(
			self,
			photos: list[str],
			progress: Optional[Callable[[int, int], None]] = None
		) -> dict[str, dict]:
		"""Lit les étiquettes (`TAGS_TO_READ`) de plusieurs photos, par lots de 100.

		Args:
			photos (list[str]): Chemins des photos.
			progress (Optional[Callable[[int, int], None]]): Fonction appelée après chaque lot avec le nombre de photos lues et leur nombre total, ou `None`.

		Returns:
			dict[str, dict]: Entrée JSON d'ExifTool de chaque photo, par clé `path_key(photo)`. Les photos qu'ExifTool n'a pas pu lire sont absentes du résultat.
		"""
		result: dict[str, dict] = {}
		for start in range(0, len(photos), 100):
			chunk = photos[start:start + 100]
			completed = self.run(["-json", "-struct", *TAGS_TO_READ], [Path(p).as_posix() for p in chunk])
			try:
				entries: list[dict] = json.loads(completed.stdout.decode("utf-8", errors = "replace") or "[]")
			except json.JSONDecodeError:
				entries = []
			for entry in entries:
				result[path_key(entry.get("SourceFile", ""))] = entry
			if progress is not None:
				progress(min(start + len(chunk), len(photos)), len(photos))
		return result

	def write_tags(self, records: list[dict], backup: bool) -> list[str]:
		"""Écrit des métadonnées dans plusieurs photos en un seul appel d'ExifTool.

		Les enregistrements passent par un fichier JSON placé dans un dossier temporaire, qui sert aussi de dossier de travail : son chemin, qui peut contenir des accents, n'est jamais passé en argument. La date de modification des fichiers est conservée.

		Args:
			records (list[dict]): Un dictionnaire par photo, avec la clé « `SourceFile` » (chemin de la photo) et les champs à écrire (« `XMP-iptcExt:PersonInImage` », « `XMP-mwg-rs:RegionInfo` »…).
			backup (bool): True pour laisser ExifTool créer une sauvegarde `*_original` de chaque photo, `False` pour modifier les photos sans copie (`-overwrite_original`).

		Returns:
			list[str]: Messages d'ExifTool, ligne par ligne (bilans, avertissements et erreurs compris).
		"""
		# Les enregistrements sont transmis dans un fichier JSON temporaire : c'est le moyen le plus fiable
		# d'écrire des structures XMP (régions) contenant des noms avec des caractères spéciaux.
		# ExifTool est lancé DANS le dossier temporaire et lit le JSON par un nom relatif : le chemin du
		# dossier temporaire (qui peut contenir des accents sous Windows) n'est ainsi jamais passé en argument.
		with tempfile.TemporaryDirectory() as directory:
			with open(os.path.join(directory, "records.json"), "w", encoding = "utf-8") as handle:
				json.dump(records, handle, ensure_ascii = False)
			options = [
				"-json=records.json",
				"-preserve",              # Conserve la date de modification des fichiers
				"-ignoreMinorErrors",
			]
			if not backup:
				options.append("-overwrite_original")
			completed = self.run(options, [record["SourceFile"] for record in records], cwd = directory)
		output = completed.stdout.decode("utf-8", errors = "replace") + completed.stderr.decode("utf-8", errors = "replace")
		return [line.strip() for line in output.splitlines() if line.strip()]


# ======================================================================================
# Détection et reconnaissance des visages (DeepFace)
# ======================================================================================

class FaceEngine:
	"""Détecte les visages d'une photo et calcule l'empreinte numérique de chacun, avec DeepFace.

	Attributes:
		deepface (ModuleType): Module DeepFace renvoyé par `load_deepface`.
		model_name (str): Modèle de reconnaissance (clé de `RECOGNITION_MODELS`).
		detector_backend (str): Détecteur de visages (élément de `DETECTOR_BACKENDS`).
		max_image_size (int): Plus grand côté, en pixels, de l'image analysée.
		min_face_size (int): Plus petit côté, en pixels, d'un visage retenu.
		detection_score (float): Confiance minimale de la détection, de `0` à `1`.
	"""

	def __init__(
			self,
			deepface: ModuleType,
			model_name: str,
			detector_backend: str,
			max_image_size: int,
			min_face_size: int,
			detection_score: float
		) -> None:
		"""Prépare DeepFace pour l'analyse des photos.

		Les deux réseaux sont construits tout de suite (et leurs poids téléchargés au premier lancement) : une erreur arrête ainsi l'analyse avant la première photo, au lieu de faire échouer les photos une à une.

		Args:
			deepface (ModuleType): Module DeepFace renvoyé par `load_deepface`.
			model_name (str): Modèle de reconnaissance (« ArcFace », « Facenet512 »…).
			detector_backend (str): Détecteur de visages (« retinaface », « yunet »…).
			max_image_size (int): Plus grand côté, en pixels, de l'image analysée : les photos plus grandes sont réduites.
			min_face_size (int): Taille minimale, en pixels, d'un visage retenu : les plus petits sont écartés.
			detection_score (float): Confiance minimale de la détection, de `0` à `1` : les visages plus incertains sont écartés.

		Raises:
			Exception: Si un modèle est inconnu ou si ses poids ne peuvent pas être téléchargés (erreur transmise par DeepFace).
		"""
		self.deepface: ModuleType = deepface
		self.model_name: str = model_name
		self.detector_backend: str = detector_backend
		self.max_image_size: int = max_image_size
		self.min_face_size: int = min_face_size
		self.detection_score: float = detection_score
		# Les deux réseaux sont construits tout de suite (et leurs poids téléchargés au premier lancement) : une erreur (pas de connexion, modèle inconnu) arrête ainsi l'analyse avant la première photo, au lieu de faire échouer les photos une à une. DeepFace les garde ensuite en mémoire pour les appels suivants.
		deepface.build_model(model_name)
		deepface.build_model(detector_backend, task = "face_detector")

	def analyse(self, photo: str) -> list[dict]:
		"""Détecte les visages d'une photo et calcule leur empreinte.

		La photo est redressée selon son orientation EXIF, puis réduite à `max_image_size`. Les visages trop petits ou trop incertains sont écartés.

		Args:
			photo (str): Chemin de la photo.

		Returns:
			list[dict]: Un dictionnaire par visage retenu, avec les clés « `box` » (`tuple[float, float, float, float]` : x, y, largeur, hauteur normalisés sur l'image redressée), « `score` » (`float` : confiance de la détection), « `embedding` » (`bytes` : empreinte en `float32`) et « `thumbnail` » (`bytes` : vignette JPEG).

		Raises:
			OSError: Si la photo ne peut pas être ouverte ou décodée.
		"""
		with Image.open(photo) as source:
			# Pour les JPEG, le décodage à échelle réduite accélère beaucoup les gros fichiers
			source.draft("RGB", (self.max_image_size, self.max_image_size))
			# On redresse l'image selon l'orientation EXIF : les coordonnées sont celles de l'image affichée
			image = ImageOps.exif_transpose(source).convert("RGB")

		# Réduction éventuelle : la détection n'a pas besoin de 24 millions de pixels
		scale = self.max_image_size / max(image.size)
		if scale < 1:
			image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)

		# DeepFace attend un tableau NumPy dans l'ordre BGR d'OpenCV (Pillow fournit du RGB). Lui passer l'image déjà décodée, plutôt que son chemin, évite aussi la lecture par OpenCV, qui échoue sur les chemins accentués sous Windows et ignore l'orientation EXIF.
		bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
		height, width = bgr.shape[:2]
		detections = self.deepface.represent(
			img_path = bgr,
			model_name = self.model_name,
			detector_backend = self.detector_backend,
			enforce_detection = False,      # Une photo sans visage n'est pas une erreur
			align = True,                   # Visage redressé d'après la position des yeux avant le calcul de l'empreinte
		)

		results: list[dict] = []
		for detection in detections:
			# Faute de visage, DeepFace renvoie l'image entière avec une confiance nulle : elle est écartée ici
			confidence = float(detection.get("face_confidence") or 0)
			if confidence <= 0 or confidence < self.detection_score:
				continue
			area = detection.get("facial_area") or {}
			try:
				x, y, w, h = (float(area[key]) for key in ("x", "y", "w", "h"))
			except (KeyError, TypeError, ValueError):
				continue
			# Rognage aux limites de l'image
			x0, y0 = max(0.0, x), max(0.0, y)
			x1, y1 = min(float(width), x + w), min(float(height), y + h)
			if min(x1 - x0, y1 - y0) < self.min_face_size:
				continue  # Visage trop petit : l'empreinte serait peu fiable
			# Empreinte de 128 à 4096 dimensions selon le modèle, stockée en flottants de 32 bits
			embedding = np.asarray(detection["embedding"], dtype = np.float32)
			results.append({
				"box": (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height),
				"score": confidence,
				"embedding": embedding.tobytes(),
				"thumbnail": encode_thumbnail(make_thumbnail(image, (x0, y0, x1, y1))),
			})
		return results


# ======================================================================================
# Modèle de données : visages, personnes, regroupement
# ======================================================================================

@dataclass(eq = False)
class FaceRecord:
	"""Un visage détecté dans une photo.

	Attributes:
		photo (str): Chemin absolu de la photo.
		box (tuple[float, float, float, float]): x, y, largeur et hauteur du visage, normalisés sur l'image redressée.
		score (float): Confiance de la détection, de `0` à `1`.
		embedding (np.ndarray): Empreinte numérique du visage (`float32`, de `128` à `4096` dimensions selon le modèle).
		face_id (int): Identifiant du visage dans la base SQLite, où sa vignette est lue à la demande.
		label (Optional[str]): Nom issu d'une région déjà étiquetée dans la photo, ou `None`.
		person (Optional[Person]): Personne à laquelle le visage est rattaché, ou `None`.
		deleted (bool): `True` si l'utilisateur l'a écarté (faux positif).
	"""
	photo: str
	box: tuple[float, float, float, float]       # x, y, largeur, hauteur normalisés (image redressée)
	score: float
	embedding: np.ndarray
	face_id: int                                 # Identifiant dans la base SQLite (la vignette y est lue à la demande)
	label: Optional[str] = None                  # Nom issu d'une région déjà étiquetée dans la photo
	person: Optional["Person"] = None
	deleted: bool = False                        # Faux positif écarté par l'utilisateur


@dataclass(eq = False)
class Person:
	"""Un groupe de visages censés appartenir à la même personne.

	Attributes:
		identifier (int): Identifiant unique dans la photothèque, repris dans la liste des personnes.
		faces (list[FaceRecord]): Visages de la personne.
		name (Optional[str]): Nom de la personne, ou `None` tant qu'elle est inconnue.
		source (str): Origine du nom : « `tag` » (étiquette existante), « `manual` » (saisi) ou vide.
		ignored (bool): `True` si la personne est exclue de l'écriture des métadonnées.
	"""
	identifier: int
	faces: list[FaceRecord] = field(default_factory = list)
	name: Optional[str] = None
	source: str = ""                             # "tag" (étiquette existante) ou "manual" (saisie)
	ignored: bool = False                        # Personne exclue de l'écriture des métadonnées

	@property
	def photos(self) -> set[str]:
		"""set[str]: Chemins des photos où apparaît la personne, sans doublon."""
		return {face.photo for face in self.faces}

	@property
	def display_name(self) -> str:
		"""str: Nom affiché : celui de la personne, ou « Inconnu n°… » tant qu'elle n'en a pas."""
		return self.name or f"Inconnu n°{self.identifier}"


def cluster_faces(embeddings: np.ndarray, label_keys: list[Optional[str]], scores: list[float],
				threshold: float) -> list[list[int]]:
	"""Regroupe des empreintes de visages par personne.

	- Les visages déjà étiquetés sont d'abord regroupés par nom : l'étiquette fait foi.
	- Les autres rejoignent le groupe le plus proche si la similarité cosinus atteint le seuil, sinon ils fondent un nouveau groupe.
	- Les groupes devenus trop proches sont fusionnés, sauf s'ils portent des noms différents.

	Args:
		embeddings (np.ndarray): Empreintes, une ligne par visage (tableau de forme (nombre, dimension)).
		label_keys (list[Optional[str]]): Clé du nom (voir `name_key`) de chaque visage déjà étiqueté, None sinon.
		scores (list[float]): Confiance de la détection de chaque visage : les plus nets sont placés en premier.
		threshold (float): Similarité cosinus minimale pour rattacher un visage à un groupe ou fusionner deux groupes.

	Returns:
		list[list[int]]: Groupes, chacun donné par les indices de ses visages dans `embeddings`.
	"""
	count = len(embeddings)
	if count == 0:
		return []
	# Normalisation L2 : le produit scalaire de deux vecteurs devient leur similarité cosinus
	vectors = embeddings / np.maximum(np.linalg.norm(embeddings, axis = 1, keepdims = True), 1e-9)
	dimension = vectors.shape[1]

	members: list[list[int]] = []
	sums: list[np.ndarray] = []
	cluster_labels: list[Optional[str]] = []
	label_to_cluster: dict[str, int] = {}
	centroids = np.zeros((count, dimension), dtype = np.float32)

	def new_cluster(label: Optional[str]) -> int:
		"""Crée un groupe vide.

		Args:
			label (Optional[str]): Clé du nom porté par le groupe, ou `None` pour un groupe sans nom.

		Returns:
			int: Rang du nouveau groupe.
		"""
		members.append([])
		sums.append(np.zeros(dimension, dtype = np.float32))
		cluster_labels.append(label)
		return len(members) - 1

	# Visages étiquetés d'abord, puis les plus nets (meilleur score de détection) en premier
	order = sorted(range(count), key = lambda i: (label_keys[i] is None, -scores[i]))
	for i in order:
		key = label_keys[i]
		if key is not None:
			if key not in label_to_cluster:
				label_to_cluster[key] = new_cluster(key)
			target = label_to_cluster[key]
		else:
			target = -1
			if members:
				similarities = centroids[:len(members)] @ vectors[i]
				best = int(np.argmax(similarities))
				if similarities[best] >= threshold:
					target = best
			if target < 0:
				target = new_cluster(None)
		members[target].append(i)
		sums[target] += vectors[i]
		centroids[target] = sums[target] / max(float(np.linalg.norm(sums[target])), 1e-9)

	# Passes de fusion : on réunit les groupes dont les centres sont proches
	while len(members) > 1:
		size = len(members)
		matrix = centroids[:size] @ centroids[:size].T
		np.fill_diagonal(matrix, -1.0)
		labeled = [c for c in range(size) if cluster_labels[c] is not None]
		if labeled:
			matrix[np.ix_(labeled, labeled)] = -1.0  # Deux groupes nommés ne fusionnent jamais
		rows, columns = np.where(np.triu(matrix) >= threshold)
		if len(rows) == 0:
			break
		pairs = sorted(zip(rows.tolist(), columns.tolist()), key = lambda pair: -matrix[pair])
		used: set[int] = set()
		for a, b in pairs:
			if a in used or b in used:
				continue  # Chaque groupe fusionne au plus une fois par passe
			used.update((a, b))
			members[a] += members[b]
			sums[a] += sums[b]
			if cluster_labels[a] is None:
				cluster_labels[a] = cluster_labels[b]
			members[b] = []
		# Suppression des groupes vidés et recalcul des centres
		keep = [c for c in range(size) if members[c]]
		members = [members[c] for c in keep]
		sums = [sums[c] for c in keep]
		cluster_labels = [cluster_labels[c] for c in keep]
		centroids = np.array([s / max(float(np.linalg.norm(s)), 1e-9) for s in sums], dtype = np.float32)
	return members


class Library:
	"""Ensemble des visages, des personnes et des étiquettes existantes.

	Attributes:
		faces (list[FaceRecord]): Tous les visages détectés, y compris ceux écartés.
		persons (dict[int, Person]): Personnes, par identifiant.
		photo_tags (dict[str, PhotoTags]): Étiquettes déjà présentes dans chaque photo, par chemin.
		photo_errors (dict[str, str]): Description de l'erreur de chaque photo en erreur (extension incorrecte, écriture refusée par ExifTool), par chemin.
		_next_id (int): Prochain identifiant de personne à attribuer.
	"""

	def __init__(self) -> None:
		"""Crée une photothèque vide, remplie ensuite par l'analyse (`scan_worker`)."""
		self.faces: list[FaceRecord] = []
		self.persons: dict[int, Person] = {}
		self.photo_tags: dict[str, PhotoTags] = {}     # Étiquettes existantes, par photo
		self.photo_errors: dict[str, str] = {}         # Photos en erreur : cadre rouge et message dans l'aperçu
		self._next_id: int = 1

	# ---------- Association avec les étiquettes existantes ----------

	def label_faces(self) -> int:
		"""Associe aux visages détectés les noms des régions déjà présentes dans les photos.

		L'appariement est glouton, par recouvrement décroissant : un visage reçoit au plus une région, et une région au plus un visage, à condition que leur IoU atteigne `REGION_MATCH_IOU`.

		Returns:
			int: Nombre de régions nommées auxquelles aucun visage détecté ne correspond.
		"""
		by_photo: dict[str, list[FaceRecord]] = {}
		for face in self.faces:
			by_photo.setdefault(face.photo, []).append(face)
		unmatched = 0
		for photo, faces in by_photo.items():
			tags = self.photo_tags.get(photo)
			if tags is None:
				continue
			regions = [region for region in tags.regions if region["name"]]
			# Appariement glouton par recouvrement décroissant : un visage ↔ une région
			candidates = sorted(
				((iou(face.box, region["box"]), f, r) for f, face in enumerate(faces) for r, region in enumerate(regions)),
				reverse = True,
			)
			used_faces: set[int] = set()
			used_regions: set[int] = set()
			for overlap, f, r in candidates:
				if overlap < REGION_MATCH_IOU:
					break
				if f in used_faces or r in used_regions:
					continue
				faces[f].label = regions[r]["name"]
				used_faces.add(f)
				used_regions.add(r)
			unmatched += len(regions) - len(used_regions)
		return unmatched

	# ---------- Regroupement ----------

	def rebuild(self, threshold: float, use_keywords: bool) -> None:
		"""Regroupe les visages en personnes, puis les nomme d'après les étiquettes existantes.

		Les personnes précédentes sont remplacées : les noms saisis et les fusions manuelles sont perdus. Les visages écartés ne sont pas regroupés.

		Args:
			threshold (float): Seuil de similarité cosinus (voir `cluster_faces`).
			use_keywords (bool): True pour accepter aussi les mots-clés comme noms de personnes.
		"""
		self.persons.clear()
		self._next_id = 1
		active = [face for face in self.faces if not face.deleted]
		if not active:
			return
		embeddings = np.stack([face.embedding for face in active])
		label_keys = [name_key(face.label) if face.label else None for face in active]
		clusters = cluster_faces(embeddings, label_keys, [face.score for face in active], threshold)
		for indexes in clusters:
			person = self.new_person()
			for i in indexes:
				active[i].person = person
				person.faces.append(active[i])
			# Nom issu des régions étiquetées : l'orthographe la plus fréquente l'emporte
			spellings = Counter(face.label for face in person.faces if face.label)
			if spellings:
				person.name, person.source = spellings.most_common(1)[0][0], "tag"
		self._name_from_weak_tags(use_keywords)

	def _name_from_weak_tags(self, use_keywords: bool) -> None:
		"""Nomme les groupes restants à partir des noms sans position (PersonInImage, mots-clés).

		Un nom est attribué à un groupe s'il figure dans au moins 60 % de ses photos et s'il domine nettement les autres candidats. Un même nom n'est donné qu'à un seul groupe.

		Args:
			use_keywords (bool): True pour compter aussi les mots-clés comme noms candidats.
		"""
		by_photo: dict[str, list[FaceRecord]] = {}
		for face in self.faces:
			by_photo.setdefault(face.photo, []).append(face)
		taken = {name_key(p.name) for p in self.persons.values() if p.name}
		proposals: list[tuple[int, Person, str, str]] = []
		for person in self.persons.values():
			if person.name:
				continue
			photos = person.photos
			counts: Counter = Counter()
			spelling: dict[str, str] = {}
			for photo in photos:
				tags = self.photo_tags.get(photo)
				if tags is None:
					continue
				# Les noms déjà donnés à un autre visage de la photo (par région) ne sont plus candidats
				used_here = {name_key(f.label) for f in by_photo.get(photo, []) if f.label}
				for candidate in tags.weak_names(use_keywords):
					if name_key(candidate) not in used_here:
						counts[name_key(candidate)] += 1
						spelling.setdefault(name_key(candidate), candidate)
			if not counts:
				continue
			ranked = counts.most_common(2)
			best_key, best_count = ranked[0]
			second_count = ranked[1][1] if len(ranked) > 1 else 0
			if best_key not in taken and best_count / len(photos) >= 0.6 and second_count <= best_count / 2:
				proposals.append((best_count, person, best_key, spelling[best_key]))
		# Les groupes les mieux étayés sont servis en premier
		proposals.sort(key = lambda proposal: -proposal[0])
		for _, person, key, display in proposals:
			if key in taken:
				continue
			taken.add(key)
			person.name, person.source = display, "tag"

	# ---------- Édition manuelle ----------

	def new_person(self, name: Optional[str] = None, source: str = "") -> Person:
		"""Crée une personne, avec un identifiant jamais encore utilisé, et l'ajoute à la photothèque.

		Args:
			name (Optional[str]): Nom de la personne, ou `None` pour une personne inconnue.
			source (str): Origine du nom : « tag », « manual » ou vide.

		Returns:
			Person: Nouvelle personne, encore sans visage.
		"""
		person = Person(identifier = self._next_id, name = name, source = source)
		self.persons[person.identifier] = person
		self._next_id += 1
		return person

	def person_by_name(self, name: str) -> Optional[Person]:
		"""Cherche la personne qui porte un nom, sans tenir compte de la casse ni des espaces superflus.

		Args:
			name (str): Nom cherché.

		Returns:
			Optional[Person]: Personne trouvée, ou `None`.
		"""
		key = name_key(name)
		for person in self.persons.values():
			if person.name and name_key(person.name) == key:
				return person
		return None

	def rename(self, person: Person, name: str) -> Person:
		"""Nomme une personne ; si ce nom est déjà porté par une autre, les deux groupes sont fusionnés.

		Args:
			person (Person): Personne à nommer.
			name (str): Nouveau nom (les espaces superflus sont retirés).

		Returns:
			Person: Personne qui porte désormais le nom : `person` elle-même, ou celle qui le portait déjà et dans laquelle `person` a été fusionnée.
		"""
		name = normalize_name(name)
		existing = self.person_by_name(name)
		if existing is not None and existing is not person:
			return self.merge([existing, person], keep = existing)
		person.name, person.source = name, "manual"
		return person

	def merge(self, persons: list[Person], keep: Optional[Person] = None) -> Person:
		"""Fusionne plusieurs personnes en une seule.

		Les personnes absorbées sont vidées de leurs visages et retirées de la photothèque.

		Args:
			persons (list[Person]): Personnes à fusionner.
			keep (Optional[Person]): Personne à conserver ; None pour choisir en priorité une personne nommée, puis celle qui a le plus de visages.

		Returns:
			Person: Personne conservée, qui réunit tous les visages.
		"""
		if keep is None:
			keep = max(persons, key = lambda p: (p.name is not None, len(p.faces)))
		for other in persons:
			if other is keep:
				continue
			for face in other.faces:
				face.person = keep
				keep.faces.append(face)
			other.faces = []
			self.persons.pop(other.identifier, None)
		return keep

	def move_face(self, face: FaceRecord, target: Person) -> None:
		"""Déplace un visage vers une autre personne.

		Une personne qui n'a plus de visage est retirée de la photothèque.

		Args:
			face (FaceRecord): Visage à déplacer.
			target (Person): Personne de destination.
		"""
		source = face.person
		if source is target:
			return
		if source is not None:
			source.faces.remove(face)
			if not source.faces:
				self.persons.pop(source.identifier, None)
		face.person = target
		target.faces.append(face)

	def delete_face(self, face: FaceRecord) -> None:
		"""Écarte un visage (faux positif) : il n'est plus affiché ni écrit dans les métadonnées.

		Une personne qui n'a plus de visage est retirée de la photothèque.

		Args:
			face (FaceRecord): Visage à écarter.
		"""
		face.deleted = True
		source = face.person
		if source is not None:
			source.faces.remove(face)
			if not source.faces:
				self.persons.pop(source.identifier, None)
		face.person = None

	def faces_of_photo(self, photo: str) -> list[FaceRecord]:
		"""Liste les visages d'une photo, sauf ceux écartés par l'utilisateur.

		Args:
			photo (str): Chemin de la photo.

		Returns:
			list[FaceRecord]: Visages de la photo.
		"""
		return [face for face in self.faces if face.photo == photo and not face.deleted]


# ======================================================================================
# Construction des métadonnées à écrire
# ======================================================================================

def build_region_info(
		tags: PhotoTags,
		items: list[tuple[str, FaceRecord]]
	) -> tuple[dict, dict[str, str]]:
	"""Construit la structure `XMP-mwg-rs:RegionInfo` en conservant les régions déjà présentes.

	Pour chaque visage nommé : si une région existante le recouvre, on la complète (nom manquant) ; si elle porte un autre nom, le choix actuel de l'utilisateur la remplace (personne renommée ou fusionnée) ; sinon on ajoute une nouvelle région « `Face` ».

	Args:
		tags (PhotoTags): Étiquettes actuelles de la photo.
		items (list[tuple[str, FaceRecord]]): Couples (nom à écrire, visage) de la photo.

	Returns:
		tuple[dict, dict[str, str]]: Structure `RegionInfo` à écrire (régions MWG : X et Y au centre, valeurs normalisées) et noms remplacés {ancien nom: nouveau nom}.
	"""
	existing = tags.region_info or {}
	region_list = [dict(region) for region in (existing.get("RegionList") or []) if isinstance(region, dict)]
	info = {
		"AppliedToDimensions": existing.get("AppliedToDimensions") or {"W": 1, "H": 1, "Unit": "pixel"},
		"RegionList": region_list,
	}

	def region_box(region: dict) -> Optional[tuple]:
		"""Convertit la zone d'une région MWG, dont X et Y désignent le centre, en rectangle.

		Args:
			region (dict): Région MWG, avec sa clé « Area » (X, Y, W, H).

		Returns:
			Optional[tuple]: Rectangle (x, y, largeur, hauteur), (x, y) étant le coin supérieur gauche ; `None` si la zone est absente ou illisible.
		"""
		area = region.get("Area") or {}
		try:
			width, height = float(area["W"]), float(area["H"])
			return (float(area["X"]) - width / 2, float(area["Y"]) - height / 2, width, height)
		except (KeyError, TypeError, ValueError):
			return None

	replacements: dict[str, str] = {}
	for name, face in items:
		x, y, width, height = face.box
		match: Optional[dict] = None
		for region in region_list:
			box = region_box(region)
			if box is not None and str(region.get("Type", "Face")) == "Face" and iou(box, face.box) >= REGION_MATCH_IOU:
				match = region
				break
		if match is not None:
			current = str(match.get("Name", "")).strip()
			if not current:
				match["Name"] = name
			elif name_key(current) != name_key(name):
				replacements[current] = name
				match["Name"] = name
			continue
		region_list.append({
			"Area": {
				"X": round(x + width / 2, 6),
				"Y": round(y + height / 2, 6),
				"W": round(width, 6),
				"H": round(height, 6),
				"Unit": "normalized",
			},
			"Name": name,
			"Type": "Face",
		})
	return info, replacements


def check_written(tags: PhotoTags, names: list[str], regions: bool, person: bool, keywords: bool) -> bool:
	"""Vérifie, d'après les étiquettes relues, que les noms demandés figurent bien dans la photo.

	Seuls les champs demandés sont contrôlés ; pour les mots-clés, seul `XMP-dc:Subject` l'est.

	Args:
		tags (PhotoTags): Étiquettes relues dans la photo.
		names (list[str]): Noms qui doivent y figurer.
		regions (bool): `True` pour contrôler les régions de visages.
		person (bool): `True` pour contrôler `XMP-iptcExt:PersonInImage`.
		keywords (bool): `True` pour contrôler les mots-clés `XMP-dc:Subject`.

	Returns:
		bool: `True` si chaque champ contrôlé contient tous les noms (et donc toujours `True` si aucun ne l'est).
	"""
	wanted = {name_key(name) for name in names}
	present: list[set[str]] = []
	if person:
		present.append({name_key(name) for name in tags.person_in_image})
	if keywords:
		present.append({name_key(name) for name in tags.subject})
	if regions:
		present.append({name_key(region["name"]) for region in tags.regions})
	return all(wanted <= found for found in present)


def face_is_tagged(face: FaceRecord, name: Optional[str], tags: Optional[PhotoTags]) -> bool:
	"""Indique si un visage est déjà marqué dans sa photo : une région qui le recouvre y porte déjà le nom de sa personne.

	Le contrôle se fait sur les étiquettes connues de la photo (relues après chaque écriture), et non sur `FaceRecord.label`, fixé à l'analyse : un visage proposé devient ainsi « marqué » dès que la validation de sa personne a écrit son nom. Un visage dont la région porte un autre nom (visage déplacé, personne renommée) reste « proposé » : l'écriture remplacerait ce nom.

	Args:
		face (FaceRecord): Visage à contrôler.
		name (Optional[str]): Nom de la personne à laquelle le visage est rattaché, ou `None` si elle est encore inconnue.
		tags (Optional[PhotoTags]): Étiquettes connues de la photo du visage, ou `None` si elles n'ont pas été lues.

	Returns:
		bool: `True` si une région nommée de la photo porte ce nom (casse et espaces superflus mis à part, voir `name_key`) et recouvre le visage d'au moins `REGION_MATCH_IOU` ; toujours `False` pour une personne sans nom.
	"""
	if not name or tags is None:
		return False
	key = name_key(name)
	return any(
		region["name"] and name_key(region["name"]) == key and iou(face.box, region["box"]) >= REGION_MATCH_IOU
		for region in tags.regions
	)


def photos_missing_name(
		library: "Library",
		person: Person,
		regions: bool,
		person_in_image: bool,
		keywords: bool
	) -> list[str]:
	"""Liste les photos de la personne où son nom manque encore, d'après les étiquettes connues et les champs à écrire.

	Sert à la fois à la colonne « Métadonnées » de la liste et à la validation : seules ces photos sont modifiées, les autres (déjà à jour) ne sont pas réécrites inutilement.

	Args:
		library (Library): Photothèque, qui fournit les étiquettes connues de chaque photo.
		person (Person): Personne concernée.
		regions (bool): `True` si les régions de visages doivent être écrites.
		person_in_image (bool): `True` si `XMP-iptcExt:PersonInImage` doit être écrit.
		keywords (bool): `True` si les mots-clés doivent être écrits.

	Returns:
		list[str]: Chemins triés des photos à écrire ; liste vide si la personne n'a pas de nom.
	"""
	if not person.name:
		return []
	return sorted(
		photo for photo in person.photos
		if not check_written(library.photo_tags.get(photo, PhotoTags()), [person.name], regions, person_in_image, keywords)
	)


# ======================================================================================
# Interface graphique
# ======================================================================================

def screen_scale(window: tk.Misc) -> float:
	"""Mesure le facteur d'agrandissement de l'écran : 1.0 à 96 ppp (100 %), 1.25 à 120 ppp (125 %), 2.0 à 200 %…

	Jamais inférieur à 1 : macOS annonce 72 ppp (la haute résolution Retina y est gérée par le système), ce qui rapetisserait l'interface au lieu de l'adapter.

	Args:
		window (tk.Misc): Fenêtre (ou widget) dont on mesure l'écran.

	Returns:
		float: Facteur d'agrandissement, arrondi au centième et au moins égal à `1.0`.
	"""
	return max(1.0, round(window.winfo_fpixels("1i") / 96, 2))


def scaled(pixels: int) -> int:
	"""Convertit une taille prévue pour un écran à 100 % en pixels réels, selon l'échelle de l'écran.

	Args:
		pixels (int): Taille, en pixels, pour un écran à 100 % (96 ppp).

	Returns:
		int: Taille agrandie selon `SETTINGS["scale"]`, arrondie au pixel.
	"""
	return round(pixels * SETTINGS["scale"])


def work_area(window: tk.Misc) -> tuple[int, int, int, int]:
	"""Donne la zone utile de l'écran principal, barre des tâches exclue.

	Ailleurs que sous Windows, c'est l'écran entier : le gestionnaire de fenêtres y replace lui-même les fenêtres.

	Args:
		window (tk.Misc): Fenêtre (ou widget) qui sert à mesurer l'écran hors de Windows.

	Returns:
		tuple[int, int, int, int]: (gauche, haut, largeur, hauteur) de la zone, en pixels.
	"""
	if sys.platform == "win32":
		rect = (ctypes.c_long * 4)()        # Structure RECT : gauche, haut, droite, bas
		try:
			if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):   # SPI_GETWORKAREA
				return rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
		except (AttributeError, OSError):
			pass
	return 0, 0, window.winfo_screenwidth(), window.winfo_screenheight()


def thumbnail_photo(
		data: Optional[bytes],
		size: int = THUMBNAIL_SIZE,
		border: Optional[str] = None
	) -> ImageTk.PhotoImage:
	"""Crée l'image `Tk` d'une vignette de la base, à la taille voulue agrandie selon l'échelle de l'écran.

	Les vignettes restent stockées en 96 pixels : les régénérer plus grandes obligerait à ré-analyser toute la photothèque, pour un gain de netteté minime.

	Le cadre éventuel est dessiné dans l'image plutôt qu'autour du widget : les thèmes natifs de `ttk` (`vista`, `aqua`) ignorent la couleur de bordure des étiquettes, et le fond du widget reste libre pour marquer la sélection.

	Args:
		data (Optional[bytes]): Vignette JPEG lue dans la base, ou `None` (un carré gris la remplace).
		size (int): Côté voulu, en pixels, pour un écran à 100 %.
		border (Optional[str]): Couleur du cadre (`#rrggbb`, voir `FACE_FRAME_COLORS`), ou `None` pour une vignette sans cadre.

	Returns:
		ImageTk.PhotoImage: Image prête pour un widget `Tk` ; il faut en garder une référence, sinon elle disparaît.
	"""
	image = decode_thumbnail(data)
	display_size = scaled(size)
	# Le cadre est pris sur le côté voulu : la vignette encadrée garde la taille prévue, et le visage reste entier (réduit, pas rogné)
	frame = scaled(FACE_FRAME_WIDTH) if border else 0
	inner = display_size - 2 * frame
	if image.size != (inner, inner):
		image = image.resize((inner, inner), Image.Resampling.LANCZOS)
	if frame:
		image = ImageOps.expand(image, border = frame, fill = border)
	return ImageTk.PhotoImage(image)


def set_title_bar_theme(window: tk.Misc) -> None:
	"""Assombrit la barre de titre d'une fenêtre en mode sombre (Windows 10 version 2004 et suivantes, Windows 11).

	La barre de titre est dessinée par Windows et non par `Tk` : c'est l'attribut DWM « mode sombre immersif » de la fenêtre qui la passe en sombre. La fenêtre doit déjà exister côté Windows, d'où `update_idletasks()`. Sans effet hors de Windows ou en thème clair.

	Args:
		window (tk.Misc): Fenêtre principale ou boîte de dialogue à assombrir.
	"""
	if sys.platform != "win32" or not SETTINGS["dark"]:
		return
	try:
		window.update_idletasks()
		# winfo_id() désigne l'intérieur de la fenêtre ; la barre de titre appartient au cadre qui l'entoure
		handle = ctypes.windll.user32.GetParent(window.winfo_id())
		value = ctypes.c_int(1)
		# DWMWA_USE_IMMERSIVE_DARK_MODE vaut 20, mais 19 dans les versions de Windows 10 antérieures à 2004
		for attribute in (20, 19):
			if ctypes.windll.dwmapi.DwmSetWindowAttribute(handle, attribute, ctypes.byref(value), ctypes.sizeof(value)) == 0:
				break
	except (AttributeError, OSError):
		pass


def apply_dark_theme(root: tk.Tk) -> None:
	"""Applique le thème sombre à toute l'application.

	Les thèmes `ttk` natifs de Windows (« vista », « winnative ») dessinent les widgets avec les styles visuels du système, toujours clairs : leurs couleurs ne se règlent pas. Le thème « clam », dessiné par `Tk` lui-même, accepte toutes les couleurs. Les widgets `Tk` classiques (canevas, texte, menus, fenêtres de dialogue) prennent les leurs dans la base d'options, qui ne vaut que pour les widgets créés ensuite : à appeler avant de construire l'interface.

	Args:
		root (tk.Tk): Fenêtre principale de l'application.
	"""
	colors = DARK_COLORS
	style = ttk.Style(root)
	style.theme_use("clam")
	# Réglages hérités par tous les widgets ttk. Dans « clam », lightcolor et darkcolor sont les deux teintes du
	# relief des bordures : identiques au fond, elles donnent un rendu plat, comme sous Windows 11.
	style.configure(
		".",
		background = colors["background"],
		foreground = colors["text"],
		fieldbackground = colors["surface"],
		bordercolor = colors["border"],
		lightcolor = colors["background"],
		darkcolor = colors["background"],
		troughcolor = colors["surface"],
		selectbackground = colors["selection"],
		selectforeground = colors["selection_text"],
		insertcolor = colors["text"],
		arrowcolor = colors["text"],
		focuscolor = colors["accent"],
	)
	style.map(
		".",
		background = [("disabled", colors["background"])],
		foreground = [("disabled", colors["disabled"])],
	)
	# Éléments cliquables : plus clairs au survol, plus encore à l'appui. Les teintes du thème « clam » pour ces états sont claires et doivent toutes être remplacées.
	for name, normal, hover, pressed in (
		("TButton", colors["button"], colors["hover"], colors["pressed"]),
		("Treeview.Heading", colors["button"], colors["hover"], colors["pressed"]),
		("TScrollbar", colors["thumb"], colors["thumb_hover"], colors["thumb_pressed"]),
		("TScale", colors["thumb"], colors["thumb_hover"], colors["thumb_pressed"]),
	):
		style.configure(
			name,
			background = normal,
			lightcolor = normal,
			darkcolor = normal,
		)
		style.map(
			name,
			background = [("disabled", colors["background"]), ("pressed", pressed), ("active", hover)],
			lightcolor = [("pressed", pressed), ("active", hover)],
			darkcolor = [("pressed", pressed), ("active", hover)],
		)
	style.configure(
		"TScrollbar",
		troughcolor = colors["background"],
		bordercolor = colors["background"]
	)
	style.configure(
		"TCheckbutton",
		indicatorbackground = colors["surface"],
		indicatorforeground = colors["text"],
	)
	style.map(
		"TCheckbutton",
		background = [("active", colors["background"])],
		indicatorbackground = [("pressed", colors["pressed"]), ("selected", colors["selection"])],
		indicatorforeground = [("selected", colors["selection_text"])],
	)
	# Liste déroulante en lecture seule : « clam » la colorie entièrement quand elle a le focus
	style.configure("TCombobox", background = colors["button"])
	style.map(
		"TCombobox",
		fieldbackground = [("readonly", colors["surface"])],
		background = [("active", colors["hover"])],
		foreground = [("disabled", colors["disabled"])],
	)
	style.configure(
		"Treeview",
		background = colors["surface"],
		foreground = colors["text"],
	)
	style.map(
		"Treeview",
		background = [("selected", colors["selection"])],
		foreground = [("selected", colors["selection_text"])],
	)
	style.configure(
		"TProgressbar",
		background = colors["accent"],
		lightcolor = colors["accent"],
		darkcolor = colors["accent"],
		bordercolor = colors["border"],
	)

	# Widgets Tk classiques, par la base d'options
	root.configure(background = colors["background"])
	for pattern, value in (
		("*Toplevel.background", colors["background"]),                 # Fenêtres de dialogue
		("*Canvas.background", colors["background"]),                   # Zone des vignettes
		("*Text.background", colors["surface"]),                        # Journal
		("*Text.foreground", colors["text"]),
		("*Text.insertBackground", colors["text"]),
		("*Text.selectBackground", colors["selection"]),
		("*Text.selectForeground", colors["selection_text"]),
		("*Text.relief", "flat"),
		("*Text.highlightThickness", 1),
		("*Text.highlightBackground", colors["border"]),
		("*Text.highlightColor", colors["border"]),
		("*Menu.background", colors["surface"]),                        # Menu contextuel des visages
		("*Menu.foreground", colors["text"]),
		("*Menu.activeBackground", colors["selection"]),
		("*Menu.activeForeground", colors["selection_text"]),
		("*TCombobox*Listbox.background", colors["surface"]),           # Liste ouverte d'une Combobox
		("*TCombobox*Listbox.foreground", colors["text"]),
		("*TCombobox*Listbox.selectBackground", colors["selection"]),
		("*TCombobox*Listbox.selectForeground", colors["selection_text"]),
	):
		root.option_add(pattern, value)


class ToolTip:
	"""Infobulle : courte aide affichée quand la souris s'arrête sur un widget.

	Tkinter n'en propose pas. Le texte est fixe, ou calculé d'après la position de la souris par une fonction qui reçoit l'événement et renvoie le texte (`None` : pas d'infobulle) ; la liste des personnes s'en sert pour décrire la colonne ou la ligne survolée. Comme sous Windows, l'infobulle n'apparaît qu'une fois la souris immobile, et un clic la masque jusqu'à ce que la souris quitte l'élément (ou passe sur une autre ligne).

	Attributes:
		widget (tk.Misc): Widget auquel l'infobulle est attachée.
		text (str | Callable[[tk.Event], Optional[str]]): Texte fixe, ou fonction qui le calcule d'après l'événement.
		window (Optional[tk.Toplevel]): Fenêtre de l'infobulle affichée, ou `None`.
		timer (Optional[str]): Minuteur d'apparition (identifiant renvoyé par `after`), ou `None`.
		current (Optional[str]): Texte affiché, ou en attente d'affichage.
		silenced (Optional[str]): Texte masqué par un clic, jusqu'à ce que la souris quitte l'élément.
		position (tuple[int, int]): Position de la souris à l'écran, en pixels, lors du dernier mouvement.
	"""

	def __init__(
			self,
			widget: tk.Misc,
			text: str | Callable[[tk.Event], Optional[str]]
		) -> None:
		"""Attache l'infobulle à un widget, sans remplacer ses autres liaisons d'événements.

		Args:
			widget (tk.Misc): Widget à décrire.
			text (str | Callable[[tk.Event], Optional[str]]): Texte fixe, ou fonction qui reçoit l'événement de la souris et renvoie le texte (`None` : pas d'infobulle à cet endroit).
		"""
		self.widget: tk.Misc = widget
		self.text: str | Callable[[tk.Event], Optional[str]] = text
		self.window: Optional[tk.Toplevel] = None
		self.timer: Optional[str] = None          # Minuteur d'apparition (identifiant renvoyé par after)
		self.current: Optional[str] = None        # Texte affiché, ou en attente d'affichage
		self.silenced: Optional[str] = None       # Texte masqué par un clic, jusqu'à la sortie de la souris
		self.position: tuple[int, int] = (0, 0)                    # Position de la souris à l'écran
		# add = "+" : les liaisons déjà posées sur le widget (menu contextuel, molette…) sont conservées. Les
		# clics sont liés bouton par bouton : pour un même widget, Tk n'exécute que la liaison la plus précise, et
		# un <ButtonPress> générique serait ignoré au profit d'un <Button-3> existant.
		for sequence in ("<Enter>", "<Motion>"):
			widget.bind(sequence, self.on_motion, add = "+")
		widget.bind("<Leave>", self.on_leave, add = "+")
		for sequence in ("<Button-1>", "<Button-2>", "<Button-3>"):
			widget.bind(sequence, self.on_press, add = "+")
		# Molette (Windows, macOS, puis Linux) et clavier : le contenu sous la souris peut changer
		for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<KeyPress>"):
			widget.bind(sequence, self.hide, add = "+")
		# Un widget détruit (changement de page, fermeture d'une boîte de dialogue) ne doit pas laisser de
		# minuteur derrière lui : son rappel désignerait une commande Tcl déjà supprimée
		widget.bind("<Destroy>", self.hide, add = "+")

	def text_at(self, event: tk.Event) -> Optional[str]:
		"""Calcule le texte de l'infobulle à l'endroit survolé.

		Args:
			event (tk.Event): Événement de la souris.

		Returns:
			Optional[str]: Texte à afficher, ou None (ou vide) s'il n'y a pas d'infobulle à cet endroit.
		"""
		return self.text(event) if callable(self.text) else self.text

	def on_motion(self, event: tk.Event) -> None:
		"""Relance le minuteur à chaque mouvement : l'infobulle n'apparaît qu'une fois la souris immobile.

		Args:
			event (tk.Event): Événement `<Enter>` ou `<Motion>`.
		"""
		text = self.text_at(event)
		if text is not None and text == self.silenced:
			return              # Clic récent sur cet élément : pas d'infobulle avant la sortie de la souris
		self.silenced = None
		if self.window is not None and text == self.current:
			return              # Infobulle déjà affichée pour ce texte : elle reste en place
		self.hide()
		if text:
			self.current = text
			self.position = (event.x_root, event.y_root)
			self.timer = self.widget.after(TOOLTIP_DELAY, self.show)

	def on_leave(self, event: tk.Event) -> None:
		"""Ferme l'infobulle quand la souris quitte le widget ; elle pourra réapparaître au retour.

		Args:
			event (tk.Event): Événement `<Leave>` (inutilisé).
		"""
		self.silenced = None
		self.hide()

	def on_press(self, event: tk.Event) -> None:
		"""Masque l'infobulle après un clic, jusqu'à ce que la souris quitte l'élément cliqué.

		Args:
			event (tk.Event): Événement du clic.
		"""
		self.silenced = self.text_at(event)
		self.hide()

	def hide(self, event: Optional[tk.Event] = None) -> None:
		"""Annule l'apparition prévue et ferme l'infobulle affichée.

		Args:
			event (Optional[tk.Event]): Événement déclencheur (molette, touche, destruction du widget), inutilisé ; `None` lors d'un appel direct.
		"""
		if self.timer is not None:
			self.widget.after_cancel(self.timer)
			self.timer = None
		if self.window is not None:
			self.window.destroy()
			self.window = None
		self.current = None

	def show(self) -> None:
		"""Ouvre l'infobulle sous la souris, sans déborder de l'écran."""
		self.timer = None
		if not self.current or not self.widget.winfo_exists():
			return
		colors = TOOLTIP_COLORS["dark" if SETTINGS["dark"] else "light"]
		# Fenêtre sans cadre ni barre de titre (set_title_bar_theme est donc inutile), cachée le temps de la placer
		window = tk.Toplevel(self.widget, background = colors["border"])
		window.withdraw()
		window.wm_overrideredirect(True)
		# Toujours au premier plan, comme les infobulles de Windows : sinon une fenêtre elle-même au premier plan
		# (y compris celle survolée) la recouvrirait
		window.wm_attributes("-topmost", True)
		try:
			# macOS : fenêtre d'aide, qui ne prend pas l'activation à la fenêtre survolée
			window.tk.call("::tk::unsupported::MacWindowStyle", "style", window._w, "help", "noActivates")
		except tk.TclError:
			pass
		# Ligne par ligne : avec --no-emoji, emoji() retire l'emoji de tête mais aussi l'espace qui le suit. Le
		# sélecteur de variante U+FE0F qui suit certains emojis (🏷️, 🖼️, 🖱️…) est dessiné par Tk comme un large
		# blanc (vérifié sous Windows) : sans lui, l'emoji s'affiche à l'identique.
		# Une étiquette par ligne : une ligne qui commence par 🟩, 🟦 ou 🟥 prend la couleur de son emoji, que Tk
		# dessine en noir et blanc (voir EMOJI_TEXT_COLORS). La couleur est lue avant emoji(), qui retire l'emoji.
		body = tk.Frame(window, background = colors["background"], padx = scaled(6), pady = scaled(4))
		body.pack(padx = 1, pady = 1)   # Le fond de la fenêtre, visible sur un pixel, dessine la bordure
		for line in self.current.split("\n"):
			tk.Label(
				body, text = emoji(line).replace("️", ""),
				justify = "left",
				anchor = "w",
				wraplength = scaled(TOOLTIP_WIDTH),
				font = "TkTooltipFont",
				background = colors["background"],
				foreground = line_color(line) or colors["text"],
				padx = 0,
				pady = 0,
			).pack(anchor = "w")
		window.update_idletasks()
		width, height = window.winfo_reqwidth(), window.winfo_reqheight()
		pointer_x, pointer_y = self.position
		x, y = pointer_x + scaled(12), pointer_y + scaled(20)
		# La zone utile n'est connue que pour l'écran principal : sur un autre écran, pas de correction
		left, top, area_width, area_height = work_area(self.widget)
		if left <= pointer_x < left + area_width and top <= pointer_y < top + area_height:
			x = max(left, min(x, left + area_width - width))
			if y + height > top + area_height:
				# Au-dessus du pointeur : placée sous la souris, l'infobulle ferait sortir la souris du widget,
				# donc disparaîtrait aussitôt pour réapparaître ensuite (clignotement)
				y = pointer_y - height - scaled(4)
		window.wm_geometry(f"+{x}+{y}")
		window.deiconify()
		self.window = window


def line_color(line: str) -> Optional[str]:
	"""Donne la couleur d'une ligne de texte qui commence par un emoji de couleur (voir `EMOJI_TEXT_COLORS`).

	À appeler avant `emoji()` : avec `--no-emoji`, l'emoji est retiré, mais la couleur du texte garde son sens.

	Args:
		line (str): Ligne de texte, avec ses emojis.

	Returns:
		Optional[str]: Couleur (`#rrggbb`) de l'emoji qui ouvre la ligne, ou `None` pour la couleur habituelle du texte.
	"""
	return EMOJI_TEXT_COLORS.get(line.lstrip()[:1])


def colored_segments(text: str) -> list[tuple[str, Optional[str]]]:
	"""Découpe une ligne en morceaux, chacun avec la couleur de l'emoji de couleur qui l'introduit.

	Un morceau coloré va de son emoji (🟩, 🟦, 🟥) jusqu'à la virgule, la parenthèse fermante ou l'emoji de couleur suivant, exclus : dans « (🟩 2 marqués, 🟦 3 proposés) », seuls « 🟩 2 marqués » et « 🟦 3 proposés » sont colorés.

	Args:
		text (str): Ligne de texte, avec ses emojis.

	Returns:
		list[tuple[str, Optional[str]]]: Morceaux dans l'ordre, avec leur couleur (`#rrggbb`) ou `None` pour la couleur habituelle du texte. Mis bout à bout, ils redonnent le texte.
	"""
	colors = "".join(EMOJI_TEXT_COLORS)
	segments: list[tuple[str, Optional[str]]] = []
	for match in re.finditer(f"([{colors}][^,){colors}]*)|([^{colors}]+)", text):
		segments.append((match.group(0), EMOJI_TEXT_COLORS[match.group(0)[0]] if match.group(1) else None))
	return segments


def strip_segment_emoji(text: str) -> str:
	"""Adapte un morceau de texte à l'option `--no-emoji`, sans retirer les blancs qui le séparent de ses voisins.

	`emoji()` retire aussi les blancs aux extrémités : sur un morceau de ligne (« , » entre deux morceaux colorés), les mots se colleraient. Ici, seul le blanc qui suit chaque emoji retiré disparaît.

	Args:
		text (str): Morceau de texte, avec ses emojis.

	Returns:
		str: Le texte tel quel ou, avec `--no-emoji`, sans ses emojis ni le blanc qui suit chacun.
	"""
	if SETTINGS["emoji"]:
		return text
	return re.sub(f"(?:{EMOJI_PATTERN.pattern})+ ?", "", text)


class ColoredLabel(ttk.Frame):
	"""Étiquette d'une ligne dont les morceaux introduits par 🟩, 🟦 ou 🟥 prennent la couleur de leur emoji.

	Tk dessine les emojis en noir et blanc : la couleur passe donc sur le texte. Un `ttk.Label` n'a qu'une couleur de texte : la ligne est composée d'une étiquette par morceau (voir `colored_segments`), placées côte à côte.

	Attributes:
		font (str | tuple): Police des morceaux.
		tooltip (Optional[Callable[[tk.Event], Optional[str]]]): Fonction qui calcule l'infobulle, attachée à chaque morceau, ou `None`.
		text (str): Texte affiché, avec ses emojis.
		parts (list[ttk.Label]): Étiquettes des morceaux affichés.
	"""

	def __init__(
			self,
			master: tk.Misc,
			font: str | tuple,
			tooltip: Optional[Callable[[tk.Event], Optional[str]]] = None
		) -> None:
		"""Crée une étiquette vide.

		Args:
			master (tk.Misc): Widget parent.
			font (str | tuple): Police des morceaux.
			tooltip (Optional[Callable[[tk.Event], Optional[str]]]): Fonction qui calcule l'infobulle, ou `None`. Elle est attachée à chaque morceau : la souris au-dessus d'un morceau a quitté le cadre, qui ne recevrait plus ses mouvements.
		"""
		super().__init__(master)
		self.font: str | tuple = font
		self.tooltip: Optional[Callable[[tk.Event], Optional[str]]] = tooltip
		self.text: str = ""
		self.parts: list[ttk.Label] = []

	def set_text(self, text: str) -> None:
		"""Remplace le texte affiché ; les morceaux ne sont recréés que si le texte change.

		Args:
			text (str): Nouveau texte, avec ses emojis (retirés à l'affichage avec `--no-emoji`).
		"""
		if text == self.text:
			return
		self.text = text
		for part in self.parts:
			part.destroy()
		self.parts.clear()
		for segment, color in colored_segments(text):
			part = ttk.Label(self, text = strip_segment_emoji(segment), font = self.font)
			if color:
				part.configure(foreground = color)
			part.pack(side = "left")
			if self.tooltip is not None:
				ToolTip(part, self.tooltip)
			self.parts.append(part)


def photo_caption(photo: str) -> str:
	"""Décrit une photo pour l'infobulle d'une vignette, dont le libellé est tronqué.

	Args:
		photo (str): Chemin de la photo.

	Returns:
		str: Nom du fichier puis dossier, sur deux lignes précédées chacune d'un emoji.
	"""
	return f"🖼️ {os.path.basename(photo)}\n📁 {os.path.dirname(photo)}"


class NameDialog(tk.Toplevel):
	"""Boîte de saisie d'un nom, avec liste déroulante des noms déjà connus.

	Attributes:
		result (Optional[str]): Nom saisi et validé, ou `None` si la saisie a été annulée.
		choices (list[str]): Noms proposés dans la liste déroulante.
		variable (tk.StringVar): Texte du champ de saisie.
		combobox (ttk.Combobox): Champ de saisie avec sa liste déroulante.
	"""

	def __init__(self, master: tk.Misc, title: str, prompt: str, choices: list[str], initial: str = "") -> None:
		"""Ouvre la boîte (modale) et attend sa fermeture : le nom saisi est alors dans `result`.

		Args:
			master (tk.Misc): Fenêtre parente.
			title (str): Titre de la boîte.
			prompt (str): Question affichée au-dessus du champ.
			choices (list[str]): Noms proposés dans la liste déroulante.
			initial (str): Texte de départ du champ.
		"""
		super().__init__(master)
		self.title(emoji(title))
		self.transient(master)
		self.resizable(False, False)
		self.result: Optional[str] = None
		self.choices: list[str] = choices

		ttk.Label(self, text = prompt, padding = 8).pack()
		self.variable: tk.StringVar = tk.StringVar(value = initial)
		self.combobox: ttk.Combobox = ttk.Combobox(self, textvariable = self.variable, values = choices, width = 42)
		self.combobox.pack(padx = 12)
		# La liste se restreint au fil de la frappe
		self.combobox.bind("<KeyRelease>", self.filter_choices)
		ToolTip(self.combobox, "🔤 Tapez un nom ou choisissez-le dans la liste, qui se restreint au fil de la frappe.")
		buttons = ttk.Frame(self, padding = 8)
		buttons.pack()
		accept_button = ttk.Button(buttons, text = "✅ Valider", command = self.accept)
		accept_button.pack(side = "left", padx = 4)
		ToolTip(accept_button, "✅ Enregistre ce nom (touche Entrée).")
		cancel_button = ttk.Button(buttons, text = "✖️ Annuler", command = self.destroy)
		cancel_button.pack(side = "left", padx = 4)
		ToolTip(cancel_button, "✖️ Ferme sans rien changer (touche Échap).")
		self.bind("<Return>", lambda event: self.accept())
		self.bind("<Escape>", lambda event: self.destroy())
		strip_widget_emoji(self)
		set_title_bar_theme(self)
		self.grab_set()
		self.combobox.focus_set()
		self.wait_window(self)

	def filter_choices(self, event: tk.Event) -> None:
		"""Restreint la liste déroulante aux noms qui contiennent le texte tapé, sans tenir compte de la casse.

		Args:
			event (tk.Event): Événement `<KeyRelease>` du champ (inutilisé).
		"""
		typed = self.variable.get().casefold()
		self.combobox.configure(values = [c for c in self.choices if typed in c.casefold()])

	def accept(self) -> None:
		"""Valide le nom saisi, s'il n'est pas vide, et ferme la boîte."""
		name = normalize_name(self.variable.get())
		if name:
			self.result = name
			self.destroy()
		else:
			self.bell()


class PhotoViewer(tk.Toplevel):
	"""Fenêtre qui affiche une photo entière, avec le cadre et le nom de chaque visage, et 💾 sur ceux déjà enregistrés dans ses métadonnées. L'erreur éventuelle de la photo (extension incorrecte…) est décrite au-dessus.

	Attributes:
		photo_image (ImageTk.PhotoImage): Photo annotée affichée (référence conservée pour qu'elle reste visible).
	"""

	def __init__(
			self,
			master: tk.Misc,
			library: Library,
			face: FaceRecord
		) -> None:
		"""Ouvre la photo d'un visage, annotée du cadre et du nom de chaque visage qu'elle contient.

		La photo est réduite pour tenir dans la zone utile de l'écran. Si elle ne peut pas être ouverte, un message d'erreur s'affiche et la fenêtre se referme aussitôt.

		Un visage déjà enregistré (son nom figure dans une région de la photo, voir `face_is_tagged`) porte une disquette 💾 dans le coin supérieur droit de son cadre, dont la couleur reste réservée au visage sélectionné ; une légende s'affiche alors sous la photo.

		Si la photo est en erreur (`Library.photo_errors`), la description de l'erreur s'affiche en rouge au-dessus d'elle.

		Args:
			master (tk.Misc): Fenêtre parente.
			library (Library): Photothèque, qui fournit les visages de la photo, le nom de leur personne et les erreurs des photos.
			face (FaceRecord): Visage dont on ouvre la photo, encadré en rouge (les autres le sont en vert).
		"""
		super().__init__(master)
		self.title(emoji(f"🖼️ {face.photo}"))
		try:
			with Image.open(face.photo) as source:
				image = ImageOps.exif_transpose(source).convert("RGB")
		except Exception as error:
			messagebox.showerror(emoji("❌ Erreur"), f"Impossible d'ouvrir la photo :\n{error}", parent = master)
			self.destroy()
			return
		# Taille prévue pour un écran à 100 %, agrandie selon l'échelle de l'écran sans dépasser sa zone utile
		_, _, area_width, area_height = work_area(self)
		image.thumbnail((min(scaled(1100), area_width - scaled(40)), min(scaled(780), area_height - scaled(80))))
		draw = ImageDraw.Draw(image)
		font = load_font(max(scaled(12), image.width // 55))
		frame = scaled(3)
		tags = library.photo_tags.get(face.photo)
		saved = 0
		for other in library.faces_of_photo(face.photo):
			x, y, width, height = other.box
			left, top = x * image.width, y * image.height
			right, bottom = (x + width) * image.width, (y + height) * image.height
			# Rouge pour le visage sélectionné, vert pour les autres
			color = "#ff3b30" if other is face else "#34c759"
			draw.rectangle([left, top, right, bottom], outline = color, width = frame)
			# Le nom est écrit sur un fond plein pour rester lisible sur les images chargées
			caption = other.person.display_name if other.person else "?"
			text_box = draw.textbbox((left, bottom + 2), caption, font = font)
			draw.rectangle(text_box, fill = color)
			draw.text((left, bottom + 2), caption, fill = "white", font = font)
			if face_is_tagged(other, other.person.name if other.person else None, tags):
				# Visage déjà enregistré dans les métadonnées : 💾 dans le coin supérieur droit, à l'intérieur du cadre,
				# car la couleur du cadre désigne déjà le visage sélectionné. Taille proportionnelle au visage, mais bornée.
				saved += 1
				side = round(min(max(scaled(16), (right - left) * 0.22), scaled(32)))
				padding = scaled(2)
				badge_left = max(0, round(right) - frame - side - 2 * padding)
				badge_top = round(top) + frame
				# Badge translucide (fond blanc arrondi et disquette) : le visage reste visible en dessous.
				# Collé avec son canal alpha comme masque, le badge est mélangé à la photo selon son opacité
				badge = saved_badge(side, padding)
				image.paste(badge, (badge_left, badge_top), badge)
		self.photo_image: ImageTk.PhotoImage = ImageTk.PhotoImage(image)
		error = library.photo_errors.get(face.photo)
		if error:
			# Description de l'erreur au-dessus de la photo, en rouge comme le cadre de ses vignettes : c'est la première
			# chose à voir, puisque les noms ne peuvent pas être écrits dans cette photo tant qu'elle n'est pas corrigée
			ttk.Label(
				self,
				text = emoji(f"🟥 Photo en erreur : {error}"),
				foreground = FACE_FRAME_COLORS["error"],
				wraplength = image.width - scaled(8),
				justify = "left",
				padding = 4,
			).pack(anchor = "w", fill = "x")
		ttk.Label(self, image = self.photo_image).pack()
		if saved:
			# Légende de l'icône, seulement quand elle apparaît sur la photo
			ttk.Label(self, text = emoji("💾 Disquette : visage déjà enregistré dans les métadonnées de la photo."), padding = 4).pack(anchor = "w")
		self.bind("<Escape>", lambda event: self.destroy())
		set_title_bar_theme(self)


class UnknownNamesDialog(tk.Toplevel):
	"""Parcourt les personnes non nommées, une par une, et propose de saisir leur nom.

	Attributes:
		app (FaceTaggerApp): Fenêtre principale, qui fournit la photothèque et la base.
		pending (list[Person]): Personnes à nommer, les plus fréquentes d'abord.
		index (int): Rang de la personne affichée dans `pending`.
		person (Optional[Person]): Personne affichée, ou None avant la première.
		images (list[ImageTk.PhotoImage]): Vignettes affichées (références conservées).
		header (ttk.Label): En-tête : rang de la personne, nombre de ses visages et de ses photos.
		gallery (ttk.Frame): Cadre des vignettes de la personne.
		variable (tk.StringVar): Texte du champ de saisie.
		combobox (ttk.Combobox): Champ de saisie du nom, avec la liste des noms déjà donnés.
	"""

	def __init__(self, app: "FaceTaggerApp") -> None:
		"""Prépare la liste des personnes sans nom, les plus fréquentes d'abord, puis affiche la première.

		Les personnes ignorées ne sont pas proposées.

		Args:
			app (FaceTaggerApp): Fenêtre principale de l'application.
		"""
		super().__init__(app)
		self.app: FaceTaggerApp = app
		self.title(emoji("✍️ Nommer les personnes non reconnues"))
		self.transient(app)
		# Les groupes les plus gros d'abord
		unnamed = [p for p in app.library.persons.values() if not p.name and not p.ignored]
		self.pending: list[Person] = sorted(unnamed, key = lambda p: -len(p.faces))
		self.index: int = 0
		self.person: Optional[Person] = None
		self.images: list[ImageTk.PhotoImage] = []

		self.header: ttk.Label = ttk.Label(self, padding = 8, font = ("TkDefaultFont", 11, "bold"))
		self.header.pack()
		self.gallery: ttk.Frame = ttk.Frame(self, padding = 8)
		self.gallery.pack()
		ttk.Label(self, text = "Nom de cette personne (ou nom déjà utilisé pour fusionner) :").pack()
		self.variable: tk.StringVar = tk.StringVar()
		self.combobox: ttk.Combobox = ttk.Combobox(self, textvariable = self.variable, width = 44)
		self.combobox.pack(padx = 12, pady = 4)
		self.combobox.bind("<KeyRelease>", self.filter_choices)
		ToolTip(
			self.combobox,
			"🔤 Tapez un nom ou choisissez-le dans la liste, qui se restreint au fil de la frappe.\n"
			"💡 Un nom déjà utilisé réunit cette personne et celle qui porte déjà ce nom.",
		)
		buttons = ttk.Frame(self, padding = 8)
		buttons.pack()
		for text, command, tip in (
			(
				"✅ Valider",
				self.validate,
				"✅ Donne ce nom à la personne et passe à la suivante (touche Entrée).",
			),
			(
				"⏭️ Passer",
				self.skip,
				"⏭️ Laisse cette personne sans nom pour l'instant et passe à la suivante.",
			),
			(
				"🙈 Ignorer cette personne",
				self.ignore,
				"🙈 Met cette personne de côté : elle ne sera plus proposée ni écrite dans les métadonnées.\n"
				"💡 Réversible avec « Ignorer / réactiver » dans la fenêtre principale.",
			),
			(
				"⏹️ Terminer",
				self.destroy,
				"⏹️ Ferme cette fenêtre (touche Échap) : les noms déjà saisis sont conservés.",
			),
		):
			button = ttk.Button(buttons, text = text, command = command)
			button.pack(side = "left", padx = 4)
			ToolTip(button, tip)
		self.bind("<Return>", lambda event: self.validate())
		self.bind("<Escape>", lambda event: self.destroy())
		strip_widget_emoji(self)
		set_title_bar_theme(self)
		self.grab_set()
		self.show_current()

	def known_names(self) -> list[str]:
		"""Liste les noms déjà donnés, pour la liste déroulante.

		Returns:
			list[str]: Noms, triés sans tenir compte de la casse.
		"""
		names = {p.name for p in self.app.library.persons.values() if p.name}
		return sorted(names, key = str.casefold)

	def filter_choices(self, event: tk.Event) -> None:
		"""Restreint la liste déroulante aux noms qui contiennent le texte tapé, sans tenir compte de la casse.

		Args:
			event (tk.Event): Événement `<KeyRelease>` du champ (inutilisé).
		"""
		typed = self.variable.get().casefold()
		self.combobox.configure(values = [n for n in self.known_names() if typed in n.casefold()])

	def show_current(self) -> None:
		"""Affiche la prochaine personne encore à nommer (celles déjà fusionnées ou nommées sont sautées)."""
		library = self.app.library
		while self.index < len(self.pending):
			candidate = self.pending[self.index]
			if library.persons.get(candidate.identifier) is candidate and not candidate.name and not candidate.ignored:
				break
			self.index += 1
		if self.index >= len(self.pending):
			messagebox.showinfo(emoji("🎉 Terminé"), "Toutes les personnes ont été traitées.", parent = self)
			self.destroy()
			return
		self.person = self.pending[self.index]
		self.header.configure(text = emoji(
			f"👤 Personne {self.index + 1} / {len(self.pending)} — "
			f"{len(self.person.faces)} visage{'s' if len(self.person.faces) != 1 else ''} dans {len(self.person.photos)} photo{'s' if len(self.person.photos) != 1 else ''}"
		))
		for child in self.gallery.winfo_children():
			child.destroy()
		self.images.clear()
		best = sorted(self.person.faces, key = lambda f: -f.score)[:12]
		thumbnails = self.app.database.thumbnails([face.face_id for face in best])
		for position, face in enumerate(best):
			image = thumbnail_photo(thumbnails.get(face.face_id))
			self.images.append(image)
			label = ttk.Label(self.gallery, image = image)
			label.grid(row = position // 6, column = position % 6, padx = 3, pady = 3)
			ToolTip(label, photo_caption(face.photo))
		self.variable.set("")
		self.combobox.configure(values = self.known_names())
		self.combobox.focus_set()

	def validate(self) -> None:
		"""Donne le nom saisi à la personne affichée (fusion si ce nom existe déjà), puis passe à la suivante."""
		name = normalize_name(self.variable.get())
		if not name or self.person is None:
			self.bell()
			return
		merged = self.app.library.rename(self.person, name)
		if merged is not self.person:
			self.app.log(f"🔗 Groupe fusionné avec « {merged.name} »")
		else:
			self.app.log(f"✍️ Personne nommée « {name} »")
		self.app.refresh_persons()
		self.index += 1
		self.show_current()

	def skip(self) -> None:
		"""Passe à la personne suivante sans nommer celle-ci."""
		self.index += 1
		self.show_current()

	def ignore(self) -> None:
		"""Met la personne affichée de côté (ni proposée, ni écrite), puis passe à la suivante."""
		if self.person is not None:
			self.person.ignored = True
			self.app.refresh_persons()
		self.index += 1
		self.show_current()


class FaceTaggerApp(tk.Tk):
	"""Fenêtre principale de l'application.

	Les threads de travail (analyse, écriture, restauration) ne touchent jamais Tk : ils envoient des événements par `post`, traités par `poll_events` dans le thread de l'interface.

	Attributes:
		options (argparse.Namespace): Options de la ligne de commande (voir `parse_arguments`).
		data_directory (Path): Dossier de données (modèles de DeepFace, base).
		exiftool (ExifTool): Adaptateur d'ExifTool.
		library (Library): Photothèque : visages, personnes et étiquettes.
		database (Database): Base SQLite.
		events (queue.Queue): Événements envoyés par les threads de travail à l'interface.
		cancel_event (threading.Event): Demande d'arrêt du travail en cours.
		worker (Optional[threading.Thread]): Dernier thread de travail lancé, ou None.
		current_person (Optional[Person]): Personne dont les vignettes sont affichées, ou `None`.
		icon_cache (dict[int, ImageTk.PhotoImage]): Icônes de la liste des personnes, par identifiant de visage.
		face_images (list[ImageTk.PhotoImage]): Images des vignettes de la page affichée (références conservées).
		face_labels (list[ttk.Label]): Vignettes de la page affichée, dans l'ordre.
		sorted_faces (list[FaceRecord]): Visages de la personne affichée, du plus net au moins net.
		page (int): Rang de la page affichée, à partir de `0`.
		grid_columns (int): Nombre de colonnes de la grille des vignettes.
		busy (bool): `True` pendant une analyse, une écriture ou une restauration.
		selected_faces (set[FaceRecord]): Visages sélectionnés parmi ceux de la personne affichée.
		selection_anchor (Optional[int]): Rang, dans `sorted_faces`, de la dernière vignette cliquée (pour Maj + clic).
		face_index (dict[FaceRecord, int]): Rang de chaque visage dans `sorted_faces`.
		face_widgets (dict[FaceRecord, ttk.Label]): Vignette de chaque visage de la page affichée.
		chosen_photos (list[str]): Photos choisies une à une par « Photos… » (chemins absolus).
		chosen_label (str): Libellé des photos choisies, affiché dans le champ « Dossier ».
		directory_var (tk.StringVar): Contenu du champ « Dossier ».
		threshold_var (tk.DoubleVar): Seuil de similarité réglé par le curseur.
		keywords_var (tk.BooleanVar): Case « Utiliser aussi les mots-clés comme noms candidats ».
		write_regions_var (tk.BooleanVar): Case « Écrire » des régions de visages.
		write_person_var (tk.BooleanVar): Case « Écrire » de `PersonInImage`.
		write_keywords_var (tk.BooleanVar): Case « Écrire » des mots-clés.
		backup_var (tk.BooleanVar): Case de la sauvegarde `*_original`.
		page_size_var (tk.StringVar): Nombre de vignettes par page choisi.
		status_var (tk.StringVar): Texte de la barre d'état.
		scan_button (ttk.Button): Bouton « Analyser ».
		stop_button (ttk.Button): Bouton « Arrêter ».
		threshold_label (ttk.Label): Valeur du seuil, affichée à côté du curseur.
		regroup_button (ttk.Button): Bouton « Regrouper à nouveau ».
		tree (ttk.Treeview): Liste des personnes.
		heading_tips (dict[str, str]): Infobulle de chaque en-tête de la liste, par identifiant de colonne.
		person_title (ColoredLabel): Titre de la personne affichée, coloré comme les cadres des vignettes.
		first_button (ttk.Button): Bouton de la première page.
		previous_button (ttk.Button): Bouton de la page précédente.
		page_label (ttk.Label): Numéro de la page affichée et nombre de pages.
		next_button (ttk.Button): Bouton de la page suivante.
		last_button (ttk.Button): Bouton de la dernière page.
		range_label (ttk.Label): Rangs des vignettes de la page affichée.
		page_size_box (ttk.Combobox): Choix du nombre de vignettes par page.
		selection_label (ttk.Label): Nombre de vignettes sélectionnées.
		select_all_button (ttk.Button): Bouton « Tout sélectionner ».
		selection_buttons (list[ttk.Button]): Boutons qui agissent sur les vignettes sélectionnées.
		canvas (tk.Canvas): Zone défilante des vignettes.
		faces_frame (ttk.Frame): Cadre des vignettes, placé dans `canvas`.
		validate_button (ttk.Button): Bouton « Valider la personne ».
		write_button (ttk.Button): Bouton « Tout écrire… ».
		restore_button (ttk.Button): Bouton « Restaurer les sauvegardes… ».
		progress (ttk.Progressbar): Barre de progression.
		log_text (tk.Text): Journal.
	"""

	def __init__(self, options: argparse.Namespace) -> None:
		"""Crée la fenêtre principale d'après les options de la ligne de commande.

		Déclare l'identité de l'application et la gestion des écrans à haute résolution, applique le thème, ouvre la base, cherche ExifTool, construit les widgets, puis lance la lecture périodique des événements.

		Args:
			options (argparse.Namespace): Options lues par `parse_arguments` et complétées par `main` (seuil, chemin de la base).

		Raises:
			RuntimeError: Si la base a été créée par une version plus récente de Face Tagger.
		"""
		# Avant super().__init__() : l'identité dans la barre des tâches et la gestion de la résolution de l'écran
		# doivent précéder la création de la première fenêtre
		set_windows_app_id()
		enable_dpi_awareness()
		super().__init__()
		self.options: argparse.Namespace = options
		SETTINGS["scale"] = screen_scale(self)
		if SETTINGS["dark"]:
			apply_dark_theme(self)
		self.title(emoji("👥 Face Tagger — visages par personne"))
		# Icône de la fenêtre, et par défaut de toutes les boîtes de dialogue (Toplevel). Linux et macOS refusent
		# les fichiers .ico (TclError) : sans icône, l'application démarre quand même, de même si le fichier est
		# introuvable.
		try:
			icon_file = resource_path("face_tagger.ico")
			self.iconbitmap(default = icon_file)
		except tk.TclError:
			pass
		# Taille prévue pour un écran à 100 %, agrandie selon l'échelle, mais limitée à la zone utile de l'écran
		# (barre des tâches exclue) et centrée dedans. La hauteur demandée à Tk exclut la barre de titre.
		left, top, area_width, area_height = work_area(self)
		title_bar = scaled(32)
		width = min(scaled(1300), area_width - scaled(16))
		height = min(scaled(820), area_height - title_bar - scaled(16))
		self.geometry(f"{width}x{height}+{left + (area_width - width) // 2}+{top + (area_height - height - title_bar) // 2}")
		self.data_directory: Path = Path(options.data_directory).expanduser()
		self.exiftool: ExifTool = ExifTool(options.exiftool)
		self.library: Library = Library()
		self.database: Database = Database(Path(options.database).expanduser())
		self.events: queue.Queue = queue.Queue()          # Messages des threads de travail vers l'interface
		self.cancel_event: threading.Event = threading.Event()
		self.worker: Optional[threading.Thread] = None
		self.current_person: Optional[Person] = None
		# Références aux images Tk (sinon le ramasse-miettes les supprime et elles disparaissent)
		self.icon_cache: dict[int, ImageTk.PhotoImage] = {}      # Icônes de la liste des personnes, par visage
		self.face_images: list[ImageTk.PhotoImage] = []          # Vignettes de la page affichée
		self.face_labels: list[ttk.Label] = []
		self.sorted_faces: list[FaceRecord] = []                 # Visages de la personne affichée, du plus net au moins net
		self.page: int = 0
		self.grid_columns: int = 0
		self.busy: bool = False                                        # Analyse, écriture ou restauration en cours
		# Vignettes sélectionnées (plusieurs possibles) parmi les visages de la personne affichée
		self.selected_faces: set[FaceRecord] = set()
		self.selection_anchor: Optional[int] = None              # Rang de la dernière vignette cliquée (pour Maj + clic)
		self.face_index: dict[FaceRecord, int] = {}              # Rang de chaque visage dans sorted_faces
		self.face_widgets: dict[FaceRecord, ttk.Label] = {}      # Vignettes de la page affichée
		# Photos choisies une à une (« Photos… ») : elles remplacent le dossier tant que le champ affiche leur libellé
		self.chosen_photos: list[str] = []
		self.chosen_label: str = ""

		# Variables liées aux widgets
		self.directory_var: tk.StringVar = tk.StringVar(value = options.directory or "")
		self.threshold_var: tk.DoubleVar = tk.DoubleVar(value = options.threshold)
		self.keywords_var: tk.BooleanVar = tk.BooleanVar(value = False)
		self.write_regions_var: tk.BooleanVar = tk.BooleanVar(value = True)
		self.write_person_var: tk.BooleanVar = tk.BooleanVar(value = True)
		self.write_keywords_var: tk.BooleanVar = tk.BooleanVar(value = True)
		self.backup_var: tk.BooleanVar = tk.BooleanVar(value = not options.no_backup)
		self.page_size_var: tk.StringVar = tk.StringVar(value = str(options.page_size))
		self.status_var: tk.StringVar = tk.StringVar(value = emoji("👋 Choisissez un dossier de photos puis cliquez sur « Analyser »."))

		self.build_widgets()
		# La colonne « Métadonnées » dépend des cases « Écrire » : elle est recalculée à chaque changement
		for variable in (self.write_regions_var, self.write_person_var, self.write_keywords_var):
			variable.trace_add("write", lambda *arguments: self.update_person_statuses())
		strip_widget_emoji(self)
		set_title_bar_theme(self)
		self.protocol("WM_DELETE_WINDOW", self.on_close)
		self.after(100, self.poll_events)
		if not self.exiftool.available:
			self.log("⚠️ ExifTool est introuvable : la lecture et l'écriture des métadonnées sont désactivées.")
			self.log("👉 Fedora : sudo dnf install perl-Image-ExifTool — Debian/Ubuntu : sudo apt install libimage-exiftool-perl")
		stats = self.database.statistics()
		self.log(f"🗄️ Base de données : {self.database.path} ({stats['photos']} photo{'s' if stats['photos'] != 1 else ''}, {stats['faces']} visage{'s' if stats['faces'] != 1 else ''}, "
				f"{stats['backups']} sauvegarde{'s' if stats['backups'] != 1 else ''} *_original suivie{'s' if stats['backups'] != 1 else ''}).")
		self.log(f"🧠 Reconnaissance des visages : DeepFace, détecteur {options.detector}, modèle {options.model}, seuil {options.threshold:.2f}.")

	# ---------- Construction de la fenêtre ----------

	def build_widgets(self) -> None:
		"""Construit la fenêtre : dossier, réglages, liste des personnes, vignettes, écriture, progression et journal. Doit suivre `apply_dark_theme()`, qui remplit la base d'options du thème sombre.
		"""
		style = ttk.Style(self)
		style.configure("Treeview", rowheight = scaled(TREE_ROW_HEIGHT))
		# Vignettes de visages : un cadre coloré de quelques pixels marque celles qui sont sélectionnées
		colors = SELECTION_COLORS["dark" if SETTINGS["dark"] else "light"]
		style.configure("Face.TLabel", padding = scaled(3))
		style.configure(
			"SelectedFace.TLabel",
			padding = scaled(3),
			background = colors["background"],
			foreground = colors["text"],
		)

		# --- Barre du haut : dossier et lancement de l'analyse ---
		top = ttk.Frame(self, padding = 6)
		top.pack(fill = "x")
		directory_label = ttk.Label(top, text = "📁 Dossier :")
		directory_label.pack(side = "left")
		directory_entry = ttk.Entry(top, textvariable = self.directory_var)
		directory_entry.pack(side = "left", fill = "x", expand = True, padx = 6)
		for widget in (directory_label, directory_entry):
			ToolTip(widget, "📁 Dossier de photos à analyser, sous-dossiers compris.\n"
				"💡 Saisissez son chemin ou choisissez-le avec « Parcourir… ».\n"
				"🖼️ « Photos… » analyse plutôt quelques photos choisies une à une.")
		browse_button = ttk.Button(top, text = "Parcourir…", command = self.choose_directory)
		browse_button.pack(side = "left")
		ToolTip(browse_button, "📂 Choisir le dossier de photos à analyser.")
		# Sans le sélecteur de variante U+FE0F après l'emoji : Tk le dessine comme un large blanc
		photos_button = ttk.Button(top, text = "🖼 Photos…", command = self.choose_photos)
		photos_button.pack(side = "left", padx = (6, 0))
		ToolTip(
			photos_button,
			"🖼️ Choisir une ou plusieurs photos précises à analyser, au lieu d'un dossier entier.\n"
			"🖱️ Ctrl ou Maj + clic dans la boîte de dialogue pour en sélectionner plusieurs.",
		)
		self.scan_button: ttk.Button = ttk.Button(top, text = "🔍 Analyser", command = self.start_scan)
		self.scan_button.pack(side = "left", padx = 6)
		ToolTip(
			self.scan_button,
			"🔍 Détecte les visages des photos avec DeepFace, lit les noms déjà présents dans leurs métadonnées, "
			"puis regroupe les visages par personne.\n"
			"💡 Les photos déjà analysées et inchangées sont reprises de la base : un nouveau passage est rapide.\n"
			"📥 Au premier lancement, DeepFace télécharge les poids de ses modèles (≈ 255 Mo avec RetinaFace et ArcFace).",
		)
		self.stop_button: ttk.Button = ttk.Button(
			top,
			text = "⏹️ Arrêter",
			command = self.cancel_event.set,
			state = "disabled"
		)
		self.stop_button.pack(side = "left")
		ToolTip(
			self.stop_button,
			f"⏹️ Interrompt l'analyse (après la photo en cours) ou l'écriture (après le lot de {EXIFTOOL_BATCH} photos "
			"en cours). Le travail déjà fait est conservé.\n"
			"⚠️ Une restauration des sauvegardes, elle, va jusqu'au bout.",
		)

		# --- Réglages du regroupement ---
		settings = ttk.Frame(self, padding = (6, 0))
		settings.pack(fill = "x")
		threshold_title = ttk.Label(
			settings,
			text = "🎚️ Seuil de similarité :"
		)
		threshold_title.pack(side = "left")
		self.threshold_label: ttk.Label = ttk.Label(
			settings,
			text = f"{self.threshold_var.get():.2f}",
			width = 5
		)
		threshold_scale = ttk.Scale(
			settings,
			from_ = THRESHOLD_MIN,
			to = THRESHOLD_MAX,
			variable = self.threshold_var,
			length = scaled(220),
			command = lambda value: self.threshold_label.configure(text = f"{float(value):.2f}"),
		)
		threshold_scale.pack(side = "left", padx = 6)
		self.threshold_label.pack(side = "left")
		threshold_hint = ttk.Label(
			settings,
			text = "(plus bas = groupes plus larges)"
		)
		threshold_hint.pack(side = "left", padx = 4)
		# Le réglage, sa valeur et leurs libellés portent la même infobulle
		model = self.options.model
		for widget in (threshold_title, threshold_scale, self.threshold_label, threshold_hint):
			ToolTip(
				widget,
				f"🎚️ Ressemblance minimale entre deux visages pour les attribuer à la même personne (de {THRESHOLD_MIN:.2f} à {THRESHOLD_MAX:.2f}).\n"
				f"🧠 Chaque modèle a sa propre échelle : pour {model}, la valeur conseillée est {RECOGNITION_MODELS[model]:.2f}.\n"
				"⬆️ Plus haut : groupes plus sûrs, mais une même personne peut être répartie entre plusieurs groupes.\n"
				"⬇️ Plus bas : groupes plus larges, au risque de mélanger des personnes.\n"
				"💡 S'applique à la prochaine analyse, ou tout de suite avec « Regrouper à nouveau ».",
			)
		keywords_check = ttk.Checkbutton(
			settings,
			text = "Utiliser aussi les mots-clés comme noms candidats",
			variable = self.keywords_var,
		)
		keywords_check.pack(side = "left", padx = 16)
		ToolTip(
			keywords_check,
			"🏷️ En plus de PersonInImage, les mots-clés des photos (XMP, IPTC, Windows) servent à nommer les groupes : "
			"un nom est retenu s'il figure dans au moins 60 % des photos d'un groupe.\n"
			"⚠️ À cocher seulement si vos mots-clés sont des noms de personnes : un mot-clé fréquent comme "
			"« Vacances » pourrait sinon nommer un groupe.\n"
			"💡 S'applique à la prochaine analyse, ou tout de suite avec « Regrouper à nouveau ».",
		)
		self.regroup_button: ttk.Button = ttk.Button(
			settings,
			text = "🔄 Regrouper à nouveau",
			command = self.regroup
		)
		self.regroup_button.pack(side = "left")
		ToolTip(
			self.regroup_button,
			"🔄 Refait le regroupement avec le seuil et l'option ci-contre, sans ré-analyser les photos.\n"
			"⚠️ Les modifications manuelles (noms saisis, fusions, visages déplacés, personnes ignorées) sont perdues ; "
			"les noms lus dans les photos sont retrouvés.",
		)

		# --- Zone centrale : personnes à gauche, visages de la personne choisie à droite ---
		panes = ttk.PanedWindow(self, orient = "horizontal")
		panes.pack(fill = "both", expand = True, padx = 6, pady = 6)

		left = ttk.Frame(panes)
		panes.add(left, weight = 1)
		self.tree: ttk.Treeview = ttk.Treeview(left, columns = ("faces", "photos", "source", "status"), selectmode = "extended")
		self.tree.heading("#0", text = "Personne")
		self.tree.heading("faces", text = "Visages")
		self.tree.heading("photos", text = "Photos")
		self.tree.heading("source", text = "Origine du nom")
		self.tree.heading("status", text = "Métadonnées")
		self.tree.column("#0", width = scaled(200))
		self.tree.column("faces", width = scaled(60), anchor = "center")
		self.tree.column("photos", width = scaled(60), anchor = "center")
		self.tree.column("source", width = scaled(100))
		self.tree.column("status", width = scaled(100))
		tree_scroll = ttk.Scrollbar(left, orient = "vertical", command = self.tree.yview)
		self.tree.configure(yscrollcommand = tree_scroll.set)
		person_buttons = ttk.Frame(left)
		person_buttons.pack(side = "bottom", fill = "x", pady = 4)
		for text, command, tip in (
			("✏️ Renommer…", self.rename_selected,
				"✏️ Nomme la personne sélectionnée (aussi par double-clic dans la liste).\n"
				"💡 Un nom déjà utilisé réunit les deux groupes en une seule personne."),
			("🔗 Fusionner", self.merge_selected,
				"🔗 Réunit en une seule personne les groupes sélectionnés, quand une même personne a été répartie "
				"entre plusieurs groupes. Le nom conservé est celui d'un groupe nommé, le plus grand s'il y en a plusieurs.\n"
				"🖱️ Ctrl ou Maj + clic dans la liste pour sélectionner plusieurs personnes."),
			("🙈 Ignorer / réactiver", self.toggle_ignore,
				"🙈 Met de côté les personnes sélectionnées (inconnus de passage, visages mal détectés…) : elles ne sont "
				"plus proposées par « Nommer les inconnus » ni écrites dans les métadonnées.\n"
				"🔁 Un second clic les réactive."),
			("🧩 Nommer les inconnus…", self.name_unknowns,
				"🧩 Présente une à une les personnes sans nom, des plus fréquentes aux plus rares, pour les nommer rapidement."),
		):
			button = ttk.Button(person_buttons, text = text, command = command)
			button.pack(side = "left", padx = 2)
			ToolTip(button, tip)
		tree_scroll.pack(side = "right", fill = "y")
		self.tree.pack(side = "left", fill = "both", expand = True)
		self.tree.bind("<<TreeviewSelect>>", self.on_select)
		self.tree.bind("<Double-Button-1>", lambda event: self.rename_selected())
		# Infobulle selon l'endroit survolé : rôle de la colonne sur les en-têtes, résumé de la personne sur les lignes
		self.heading_tips: dict[str, str] = {
			"#0": "👤 Nom de la personne, ou « Inconnu n°… » tant qu'elle n'est pas nommée. Les personnes nommées "
				"viennent d'abord (par ordre alphabétique), puis les inconnues, des plus fréquentes aux plus rares.",
			"#1": "🙂 Nombre de visages regroupés pour cette personne.",
			"#2": "🖼️ Nombre de photos où elle apparaît.",
			"#3": "🏷️ étiquette : nom lu dans les métadonnées des photos.\n✍️ saisi : nom donné dans Face Tagger.\n"
				"❓ vide : personne pas encore nommée.",
			"#4": "💾 État des métadonnées, selon les cases « Écrire » cochées :\n"
				"✅ à jour : le nom figure déjà dans toutes ses photos.\n"
				"✏️ N à écrire : photos où le nom manque encore ; « Valider la personne » l'y inscrit.\n"
				"🙈 ignorée : jamais écrite.",
		}
		ToolTip(self.tree, self.tree_tooltip)

		right = ttk.Frame(panes)
		panes.add(right, weight = 3)
		# Morceaux colorés (🟩 marqués, 🟦 proposés, 🟥 en erreur), l'infobulle (légende des cadres) sur chacun
		self.person_title: ColoredLabel = ColoredLabel(right, font = ("TkDefaultFont", 12, "bold"), tooltip = self.person_title_tooltip)
		self.person_title.pack(anchor = "w", padx = 4)

		# Navigation entre les pages : seules les vignettes de la page affichée sont chargées en mémoire
		navigation = ttk.Frame(right)
		navigation.pack(fill = "x", padx = 4, pady = 2)
		self.first_button: ttk.Button = ttk.Button(
			navigation,
			text = "«",
			width = 3,
			command = lambda: self.go_to_page(0)
		)
		self.previous_button: ttk.Button = ttk.Button(
			navigation,
			text = "‹",
			width = 3,
			command = lambda: self.go_to_page(self.page - 1)
		)
		self.page_label: ttk.Label = ttk.Label(
			navigation,
			text = "Page 1 / 1",
			width = 14,
			anchor = "center"
		)
		self.next_button: ttk.Button = ttk.Button(
			navigation,
			text = "›",
			width = 3,
			command = lambda: self.go_to_page(self.page + 1)
		)
		self.last_button: ttk.Button = ttk.Button(
			navigation,
			text = "»",
			width = 3,
			command = lambda: self.go_to_page(10 ** 9)
		)
		for widget in (self.first_button, self.previous_button, self.page_label, self.next_button, self.last_button):
			widget.pack(side = "left", padx = 1)
		for widget, tip in (
			(self.first_button, "⏮️ Première page"),
			(self.previous_button, "⬅️ Page précédente"),
			(self.next_button, "➡️ Page suivante"),
			(self.last_button, "⏭️ Dernière page"),
		):
			ToolTip(widget, tip)
		self.range_label: ttk.Label = ttk.Label(navigation, text = "")
		self.range_label.pack(side = "left", padx = 10)
		self.page_size_box: ttk.Combobox = ttk.Combobox(
			navigation,
			textvariable = self.page_size_var, values = ("50", "100", "200", "500"),
			width = 5,
			state = "readonly",
		)
		self.page_size_box.pack(side = "right")
		page_size_label = ttk.Label(navigation, text = "Par page :")
		page_size_label.pack(side = "right", padx = 4)
		self.page_size_box.bind("<<ComboboxSelected>>", lambda event: self.go_to_page(0))
		for widget in (page_size_label, self.page_size_box):
			ToolTip(
				widget,
				"📄 Nombre de vignettes par page. Seules celles de la page affichée sont chargées : "
				"une valeur plus petite économise la mémoire et accélère l'affichage.",
			)

		# Sélection de vignettes : les commandes de cette barre s'appliquent à toutes les vignettes sélectionnées,
		# y compris celles des autres pages
		selection_bar = ttk.Frame(right)
		selection_bar.pack(fill = "x", padx = 4, pady = 2)
		self.selection_label: ttk.Label = ttk.Label(selection_bar, text = "Aucun visage sélectionné", width = 26)
		self.selection_label.pack(side = "left")
		ToolTip(
			self.selection_label,
			"☑️ Nombre de vignettes sélectionnées, toutes pages confondues.\n"
			"🖱️ Clic : sélectionner une vignette — Ctrl + clic : ajouter ou retirer — Maj + clic : sélectionner une plage.\n"
			"⌨️ Ctrl + A : tout sélectionner — Échap : tout désélectionner — Suppr : écarter.",
		)
		# Textes des boutons sans le sélecteur de variante U+FE0F, que Tk dessine comme un large blanc
		self.select_all_button: ttk.Button = ttk.Button(selection_bar, text = "☑ Tout sélectionner", command = self.select_all_faces)
		self.select_all_button.pack(side = "left", padx = 2)
		ToolTip(self.select_all_button, "☑️ Sélectionne tous les visages de la personne affichée, toutes pages confondues (Ctrl + A).")
		self.selection_buttons: list[ttk.Button] = []
		for text, command, tip in (
			("↪ Déplacer…", self.move_selected_faces,
				"↪️ Déplace les visages sélectionnés vers une personne, existante ou nouvelle, choisie par son nom.\n"
				"💡 Pour corriger d'un coup les visages attribués à tort à cette personne."),
			("✂ Détacher", self.detach_selected_faces,
				"✂️ Réunit les visages sélectionnés dans un nouveau groupe, sans nom, à nommer ensuite."),
			("🗑 Écarter", self.delete_selected_faces,
				"🗑️ Écarte les visages sélectionnés (ce ne sont pas des visages) : ils ne sont plus affichés ni écrits "
				"dans les métadonnées (touche Suppr).\n"
				"💡 Une nouvelle analyse les fait réapparaître."),
		):
			button = ttk.Button(selection_bar, text = text, command = command, state = "disabled")
			button.pack(side = "left", padx = 2)
			ToolTip(button, tip)
			self.selection_buttons.append(button)

		ttk.Label(
			right,
			text = "Clic : sélectionner — Ctrl ou Maj + clic : sélection multiple — double-clic : ouvrir la photo — clic droit : actions"
		).pack(anchor = "w", padx = 4)
		self.canvas: tk.Canvas = tk.Canvas(right, highlightthickness = 0, takefocus = 1)
		canvas_scroll = ttk.Scrollbar(
			right,
			orient = "vertical",
			command = self.canvas.yview
		)
		self.canvas.configure(yscrollcommand = canvas_scroll.set)
		canvas_scroll.pack(side = "right", fill = "y")
		self.canvas.pack(side = "left", fill = "both", expand = True)
		self.faces_frame: ttk.Frame = ttk.Frame(self.canvas)
		self.canvas.create_window((0, 0), window = self.faces_frame, anchor = "nw")
		self.faces_frame.bind(
			"<Configure>",
			lambda event: self.canvas.configure(scrollregion = self.canvas.bbox("all"))
		)
		self.canvas.bind(
			"<Configure>",
			lambda event: self.layout_faces()
		)
		# Molette de la souris (Windows / macOS : <MouseWheel>, Linux : boutons 4 et 5)
		self.canvas.bind(
			"<MouseWheel>",
			lambda event: self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
		)
		self.canvas.bind(
			"<Button-4>",
			lambda event: self.canvas.yview_scroll(-1, "units")
		)
		self.canvas.bind(
			"<Button-5>",
			lambda event: self.canvas.yview_scroll(1, "units")
		)
		# Raccourcis clavier de la sélection : le canevas prend le focus à chaque clic sur une vignette. « break »
		# empêche Tk de transmettre la touche aux liaisons générales de la fenêtre.
		for sequence, handler in (
			("<Control-a>", self.select_all_faces),
			("<Control-A>", self.select_all_faces),      # Verrouillage des majuscules actif
			("<Escape>", self.clear_face_selection),
			("<Delete>", self.delete_selected_faces),
		):
			self.canvas.bind(sequence, lambda event, handler = handler: (handler(), "break")[1])
		# Clic dans le vide, entre ou sous les vignettes : tout désélectionner, comme dans l'Explorateur
		for widget in (self.canvas, self.faces_frame):
			widget.bind("<Button-1>", lambda event: self.clear_face_selection())

		# --- Bas de fenêtre : écriture des métadonnées, progression et journal ---
		bottom = ttk.Frame(self, padding = 6)
		bottom.pack(fill = "x")
		write_label = ttk.Label(bottom, text = "💾 Écrire :")
		write_label.pack(side = "left")
		ToolTip(
			write_label,
			"💾 Métadonnées que « Valider la personne » et « Tout écrire » inscrivent dans les photos.\n"
			"📋 La colonne « Métadonnées » de la liste des personnes en tient compte.",
		)
		for text, variable, padding, tip in (
			("régions de visages (MWG)", self.write_regions_var, 4,
				"🔲 Position et nom de chaque visage (XMP-mwg-rs:RegionInfo), au format MWG reconnu par de nombreux "
				"logiciels de photos (digiKam, Lightroom…).\n"
				"💡 Une personne renommée remplace son ancien nom dans les régions existantes."),
			("PersonInImage", self.write_person_var, 4,
				"👤 Liste des personnes présentes sur la photo (XMP-iptcExt:PersonInImage), sans leur position."),
			("mots-clés (XMP, EXIF)", self.write_keywords_var, 4,
				"🏷️ Ajoute les noms aux mots-clés XMP (XMP-dc:Subject) et Windows (XPKeywords, affichés par "
				"l'Explorateur dans les propriétés du fichier).\n"
				"💡 Les mots-clés IPTC ne sont pas modifiés : leur jeu de caractères, pas toujours déclaré, "
				"risquerait d'abîmer les accents."),
			("sauvegarde *_original", self.backup_var, 12,
				"📦 Avant de modifier une photo pour la première fois, ExifTool en garde une copie « nom.jpg_original ». "
				"Les écritures suivantes ne la remplacent pas : elle reste la version d'origine.\n"
				"⚠️ Décochée : les photos sont modifiées sans copie de sauvegarde."),
		):
			check = ttk.Checkbutton(bottom, text = text, variable = variable)
			check.pack(side = "left", padx = padding)
			ToolTip(check, tip)
		self.validate_button: ttk.Button = ttk.Button(
			bottom,
			text = "✅ Valider la personne",
			command = self.validate_person,
			state = "disabled"
		)
		self.validate_button.pack(side = "right")
		ToolTip(
			self.validate_button,
			"✅ Écrit le nom de la personne sélectionnée dans celles de ses photos où il manque encore, selon les cases "
			"cochées, puis relit chaque photo pour vérifier.\n"
			"👤 Une seule personne à la fois : parcourez d'abord ses vignettes, et déplacez ou écartez les intrus.\n"
			"💡 Les étiquettes déjà présentes sont conservées, sauf l'ancien nom d'une personne renommée.",
		)
		self.write_button: ttk.Button = ttk.Button(bottom, text = "💾 Tout écrire…", command = self.start_write)
		self.write_button.pack(side = "right", padx = 6)
		ToolTip(
			self.write_button,
			"💾 Écrit d'un coup le nom de toutes les personnes nommées (et non ignorées) dans les photos où il manque encore.\n"
			"⚠️ Sans vérification personne par personne : « Valider la personne » est plus sûr.",
		)
		self.restore_button: ttk.Button = ttk.Button(bottom, text = "♻️ Restaurer les sauvegardes…", command = self.start_restore)
		self.restore_button.pack(side = "right")
		ToolTip(
			self.restore_button,
			"♻️ Remet dans leur version d'origine les photos d'un dossier (sous-dossiers compris) : chaque copie "
			"*_original reprend la place de la photo modifiée.\n"
			"⚠️ Les noms écrits dans ces photos sont perdus. Le dossier est demandé, puis une confirmation.",
		)

		self.progress: ttk.Progressbar = ttk.Progressbar(self, mode = "determinate")
		self.progress.pack(fill = "x", padx = 6)
		ttk.Label(self, textvariable = self.status_var, padding = (6, 2)).pack(fill = "x")
		# Journal : zone de texte et ascenseur ttk (celui de ScrolledText, natif sous Windows, resterait clair en
		# mode sombre)
		log_frame = ttk.Frame(self)
		log_frame.pack(fill = "x", padx = 6, pady = (0, 6))
		self.log_text: tk.Text = tk.Text(log_frame, height = 8, state = "disabled", wrap = "word")
		log_scroll = ttk.Scrollbar(log_frame, orient = "vertical", command = self.log_text.yview)
		self.log_text.configure(yscrollcommand = log_scroll.set)
		log_scroll.pack(side = "right", fill = "y")
		self.log_text.pack(side = "left", fill = "x", expand = True)

	# ---------- Journal et communication avec les threads ----------

	def log(self, message: str) -> None:
		"""Ajoute un message au journal de la fenêtre et à la console.

		À appeler depuis le thread de l'interface ; les threads de travail passent par `post("log", message)`.

		Args:
			message (str): Message à afficher (ses emojis sont retirés avec `--no-emoji`).
		"""
		# Couleur lue avant emoji(), qui retire l'emoji : Tk dessine 🟩 🟦 🟥 en noir et blanc, le texte prend leur couleur
		color = line_color(message)
		message = emoji(message)
		print(message, flush = True)
		self.log_text.configure(state = "normal")
		if color:
			self.log_text.tag_configure(color, foreground = color)     # Une étiquette Tk par couleur, nommée d'après elle
		self.log_text.insert("end", message + "\n", (color,) if color else ())
		self.log_text.see("end")
		self.log_text.configure(state = "disabled")

	def post(self, *event: object) -> None:
		"""Envoie un événement au thread de l'interface ; appelable depuis un thread de travail.

		Args:
			*event (object): Type de l'événement (« `log` », « `progress` », « `library` », « `tags` », « `restored` » ou « `done` »), suivi de ses données (voir `poll_events`).
		"""
		self.events.put(event)

	def poll_events(self) -> None:
		"""Traite les événements envoyés par les threads de travail, puis se reprogramme 100 ms plus tard.

		Événements reconnus :
		- (« `log` », message) : ajoute le message au journal ;
		- (« `progress` », fait, total, texte) : met à jour la barre de progression et la barre d'état ;
		- (« `library` », photothèque) : remplace la photothèque et reconstruit la liste des personnes ;
		- (« `tags` », `{photo: PhotoTags}`) : étiquettes relues après une écriture ;
		- (« `errors` », `{photo: Optional[str]}`) : erreurs constatées pendant une écriture (`None` : l'erreur a disparu) ;
		- (« `restored` », photos) : photos restaurées depuis leur sauvegarde ;
		- (« `done` »,) : fin du travail, les boutons sont réactivés.
		"""
		try:
			while True:
				event = self.events.get_nowait()
				kind = event[0]
				if kind == "log":
					self.log(event[1])
				elif kind == "progress":
					_, done, total, text = event
					self.progress.configure(maximum = max(total, 1), value = done)
					self.status_var.set(emoji(text))
				elif kind == "library":
					self.library = event[1]
					self.refresh_persons()
				elif kind == "tags":
					# Étiquettes relues après une écriture : la colonne « Métadonnées » et le cadre des vignettes changent
					self.library.photo_tags.update(event[1])
					self.update_person_statuses()
					self.update_face_frames()
				elif kind == "errors":
					# Erreurs constatées (ou disparues) pendant une écriture : le cadre des vignettes change
					for photo, error in event[1].items():
						if error:
							self.library.photo_errors[photo] = error
						else:
							self.library.photo_errors.pop(photo, None)
					self.update_face_frames()
				elif kind == "restored":
					self.after_restore(event[1])
				elif kind == "done":
					self.set_busy(False)
		except queue.Empty:
			pass
		self.after(100, self.poll_events)

	def set_busy(self, busy: bool) -> None:
		"""Active ou désactive les boutons pendant une analyse, une écriture ou une restauration.

		Args:
			busy (bool): True au début d'un travail, False à sa fin.
		"""
		self.busy = busy
		state = "disabled" if busy else "normal"
		for button in (self.scan_button, self.write_button, self.regroup_button, self.restore_button):
			button.configure(state = state)
		self.stop_button.configure(state = "normal" if busy else "disabled")
		self.update_action_states()

	def on_close(self) -> None:
		"""Fermeture de la fenêtre : demande l'arrêt du travail en cours, ferme la base puis la fenêtre."""
		self.cancel_event.set()
		self.database.close()
		self.destroy()

	# ---------- Analyse des photos (thread de travail) ----------

	def choose_directory(self) -> None:
		"""Choisit le dossier de photos à analyser ; il remplace les photos choisies une à une."""
		chosen = filedialog.askdirectory(initialdir = self.base_directory())
		if chosen:
			self.chosen_photos = []                  # Un dossier remplace les photos choisies une à une
			self.directory_var.set(chosen)

	def choose_photos(self) -> None:
		"""Choisit une ou plusieurs photos précises à analyser, au lieu d'un dossier entier."""
		# Extensions en minuscules et en majuscules : sous Linux, le filtre de Tk distingue les deux
		patterns = " ".join(f"*{extension} *{extension.upper()}" for extension in sorted(IMAGE_EXTENSIONS))
		chosen = filedialog.askopenfilenames(
			parent = self,
			title = "Photos à analyser",
			initialdir = self.base_directory(),
			filetypes = [("Photos", patterns), ("Tous les fichiers", "*.*")],
		)
		# splitlist : selon la version de Tk, le résultat est un tuple ou une chaîne au format Tcl
		files = [os.path.abspath(path) for path in self.tk.splitlist(chosen)] if chosen else []
		files = sorted({path for path in files if Path(path).suffix.lower() in IMAGE_EXTENSIONS})
		if not files:
			return
		self.chosen_photos = files
		folders = {os.path.dirname(path) for path in files}
		where = f"dans {folders.pop()}" if len(folders) == 1 else f"dans {len(folders)} dossiers"
		self.chosen_label = emoji(f"📑 {len(files)} photo{'s' if len(files) != 1 else ''} choisie{'s' if len(files) != 1 else ''} {where}")
		self.directory_var.set(self.chosen_label)

	def using_chosen_photos(self) -> bool:
		"""Indique si l'analyse porte sur les photos choisies une à une plutôt que sur un dossier.

		Returns:
			bool: `True` si des photos ont été choisies et que le champ « Dossier » affiche encore leur libellé (il n'a pas été modifié depuis).
		"""
		return bool(self.chosen_photos) and self.directory_var.get() == self.chosen_label

	def base_directory(self) -> str:
		"""Donne le dossier de départ des boîtes de dialogue.

		Returns:
			str: Dossier des photos choisies, sinon dossier saisi, ou à défaut dossier personnel.
		"""
		if self.using_chosen_photos():
			return os.path.dirname(self.chosen_photos[0])
		return self.directory_var.get().strip() or str(Path.home())

	def start_scan(self) -> None:
		"""Vérifie la source (dossier ou photos choisies) puis lance l'analyse dans un thread de travail."""
		if self.using_chosen_photos():
			directory: Optional[str] = None
			photos: Optional[list[str]] = [photo for photo in self.chosen_photos if os.path.isfile(photo)]
			if not photos:
				messagebox.showerror(
					emoji("❌ Photos introuvables"),
					"Les photos choisies n'existent plus. Choisissez-en d'autres."
				)
				return
		else:
			directory = os.path.abspath(os.path.expanduser(self.directory_var.get().strip()))
			photos = None
			if not os.path.isdir(directory):
				messagebox.showerror(
					emoji("❌ Dossier introuvable"),
					"Choisissez un dossier de photos valide."
				)
				return
		self.cancel_event.clear()
		self.set_busy(True)
		self.progress.configure(value = 0)
		# Les valeurs des widgets sont lues ici, dans le thread de l'interface
		threshold, use_keywords = float(self.threshold_var.get()), self.keywords_var.get()
		self.worker = threading.Thread(
			target = self.scan_worker,
			args = (directory, photos, threshold, use_keywords),
			daemon = True)
		self.worker.start()

	def create_engine(self) -> FaceEngine:
		"""Charge DeepFace et ses deux réseaux ; n'est appelée que si une photo doit être analysée.

		Exécutée dans le thread d'analyse : les messages passent par `post`.

		Returns:
			FaceEngine: Moteur prêt, réglé d'après les options de la ligne de commande.

		Raises:
			RuntimeError: Si DeepFace est introuvable.
		"""
		self.post("log", f"🧠 Chargement de DeepFace (détecteur {self.options.detector}, modèle {self.options.model})…")
		weights = self.data_directory / ".deepface" / "weights"
		if not weights.is_dir() or not any(weights.iterdir()):
			self.post("log", "📥 Premier lancement : DeepFace télécharge les poids de ses modèles, ce qui peut prendre quelques minutes.")
		deepface = load_deepface(self.data_directory)
		return FaceEngine(
			deepface,
			self.options.model,
			self.options.detector,
			self.options.max_image_size,
			self.options.min_face_size,
			self.options.detection_score,
		)

	def scan_worker(
			self,
			directory: Optional[str],
			photos: Optional[list[str]],
			threshold: float,
			use_keywords: bool
		) -> None:
		"""Détecte les visages, lit les étiquettes existantes puis regroupe les visages (exécuté hors du thread Tk).

		Ce qui est déjà en base pour une photo inchangée n'est ni ré-analysé ni relu. Le résultat est envoyé à l'interface par un événement « `library` » ; les erreurs sont écrites dans le journal, et l'événement « done » est envoyé dans tous les cas.

		Args:
			directory (Optional[str]): Dossier à parcourir (sous-dossiers compris), ou `None` si `photos` est fourni.
			photos (Optional[list[str]]): Photos choisies une à une (chemins absolus), ou `None` pour parcourir `directory`.
			threshold (float): Seuil de similarité cosinus du regroupement.
			use_keywords (bool): `True` pour accepter aussi les mots-clés comme noms de personnes.
		"""
		try:
			if photos is None:
				self.post("log", f"📂 Exploration de {directory}…")
				photos = list_photos(directory)
				if not photos:
					self.post("log", "⚠️ Aucune photo trouvée dans ce dossier.")
					return
				self.post("log", f"🖼️ {len(photos)} photo{'s' if len(photos) != 1 else ''} trouvée{'s' if len(photos) != 1 else ''}.")
			else:
				self.post("log", f"📑 {len(photos)} photo{'s' if len(photos) != 1 else ''} choisie{'s' if len(photos) != 1 else ''}.")

			# Si le moteur ou les paramètres de détection ont changé, les visages mémorisés ne sont plus valables :
			# les empreintes de deux modèles différents ne sont pas comparables entre elles
			signature = "|".join((
				"deepface",
				os.environ.get("DEEPFACE_BACKEND_ENGINE", "").strip() or DEEPFACE_ENGINE,
				self.options.model,
				self.options.detector,
				str(self.options.max_image_size),
				str(self.options.min_face_size),
				str(self.options.detection_score),
			))
			if self.database.reset_analysis_if_needed(signature):
				self.post("log", "🗄️ Modèle ou paramètres de détection modifiés : les photos seront ré-analysées.")
			known = self.database.signatures()

			engine: Optional[FaceEngine] = None
			processed: set[str] = set()
			analysed = reused = failures = 0
			for index, photo in enumerate(photos, start = 1):
				if self.cancel_event.is_set():
					self.post("log", "⏹️ Analyse interrompue : les photos déjà traitées sont conservées.")
					break
				self.post("progress", index, len(photos), f"🔍 Analyse {index}/{len(photos)} — {os.path.basename(photo)}")
				try:
					stat = os.stat(photo)
				except OSError as error:
					failures += 1
					self.post("log", f"⚠️ Photo inaccessible ({os.path.basename(photo)}) : {error}")
					continue
				previous = known.get(photo)
				if previous and previous[0] == stat.st_mtime_ns and previous[1] == stat.st_size and previous[2]:
					reused += 1                          # Photo inchangée déjà analysée : on réutilise la base
				else:
					if engine is None:
						engine = self.create_engine()    # Une erreur ici (modèles absents) arrête toute l'analyse
					try:
						found = engine.analyse(photo)
					except Exception as error:
						failures += 1
						self.post("log", f"⚠️ Photo ignorée ({os.path.basename(photo)}) : {error}")
						continue
					self.database.store_analysis(photo, stat.st_mtime_ns, stat.st_size, found)
					analysed += 1
				processed.add(photo)
			self.post("log", f"🗄️ {reused} photo{'s' if reused != 1 else ''} reprise{'s' if reused != 1 else ''} depuis la base, {analysed} analysée{'s' if analysed != 1 else ''}, {failures} en échec.")

			# Les visages viennent de la base (sans les vignettes, lues plus tard à la demande)
			library = Library()
			for face_id, photo, x, y, width, height, score, blob in self.database.load_faces(processed):
				library.faces.append(FaceRecord(
					photo = photo, box = (x, y, width, height), score = score,
					embedding = np.frombuffer(blob, dtype = np.float32), face_id = face_id,
				))
			self.post("log", f"✅ {len(library.faces)} visage{'s' if len(library.faces) != 1 else ''} au total.")

			# Contrôle du format des photos qui ont des visages (les seules affichées) : une extension qui ne
			# correspond pas au contenu ne gêne pas l'analyse, mais ExifTool refuserait d'y écrire les noms.
			# Refait à chaque analyse (seul l'en-tête est lu), car un fichier peut être remplacé sans changer de nom.
			photos_with_faces = sorted({face.photo for face in library.faces})
			for index, photo in enumerate(photos_with_faces, start = 1):
				if index % 100 == 0 or index == len(photos_with_faces):
					self.post("progress", index, len(photos_with_faces), f"🧪 Contrôle des fichiers {index}/{len(photos_with_faces)}…")
				error = photo_format_error(photo)
				if error:
					library.photo_errors[photo] = error
					self.post("log", f"🟥 {os.path.basename(photo)} : {error}")
			if library.photo_errors:
				count = len(library.photo_errors)
				self.post("log", f"⚠️ {count} photo{'s' if count != 1 else ''} en erreur (cadre rouge) : {'leurs' if count != 1 else 'ses'} noms ne seront pas écrits tant que l'erreur demeure.")

			# Étiquettes déjà présentes : d'abord depuis la base, ExifTool ne lit que les photos manquantes
			if self.exiftool.available and library.faces:
				with_faces = sorted({face.photo for face in library.faces})
				stored = self.database.tags(with_faces)
				missing = [photo for photo in with_faces if photo not in stored]
				self.post("log", f"🗄️ Étiquettes : {len(stored)} reprises depuis la base, {len(missing)} à lire avec ExifTool.")
				if missing:
					entries = self.exiftool.read_tags(
						missing, lambda done, total: self.post("progress", done, total, f"🏷️ Lecture des étiquettes {done}/{total}…"),
					)
					# On ne mémorise que les photos réellement lues : les autres seront retentées au prochain passage
					fresh = {photo: entries[path_key(photo)] for photo in missing if path_key(photo) in entries}
					self.database.store_tags(fresh)
					stored.update(fresh)
				for photo in with_faces:
					library.photo_tags[photo] = parse_photo_tags(stored.get(photo, {}))
				unmatched = library.label_faces()
				labeled = sum(1 for face in library.faces if face.label)
				self.post("log", f"🏷️ {labeled} visage{'s' if labeled != 1 else ''} rattaché{'s' if labeled != 1 else ''} à un nom déjà présent dans les photos.")
				if unmatched:
					self.post("log", f"ℹ️ {unmatched} région{'s' if unmatched != 1 else ''} nommée{'s' if unmatched != 1 else ''} sans visage détecté correspondant.")

			library.rebuild(threshold, use_keywords)
			named = sum(1 for person in library.persons.values() if person.name)
			self.post("log", f"👥 {len(library.persons)} personne{'s' if len(library.persons) != 1 else ''} regroupée{'s' if len(library.persons) != 1 else ''}, dont {named} nommée{'s' if named != 1 else ''} automatiquement.")
			self.post("library", library)
			self.post("progress", 1, 1, f"✅ Analyse terminée : {len(library.persons)} personne{'s' if len(library.persons) != 1 else ''}.")
		except Exception as error:
			self.post("log", f"❌ Erreur : {error}")
		finally:
			self.post("done")

	def regroup(self) -> None:
		"""Refait le regroupement avec le seuil courant, sans ré-analyser les photos."""
		if not self.library.faces:
			return
		if not messagebox.askyesno(
			emoji("🔄 Regrouper"),
			"Le regroupement sera refait : les noms saisis et les fusions manuelles seront perdus. Continuer ?",
		):
			return
		self.library.rebuild(float(self.threshold_var.get()), self.keywords_var.get())
		self.refresh_persons()
		self.log(f"🔄 Regroupement refait avec le seuil {self.threshold_var.get():.2f} : {len(self.library.persons)} personne{'s' if len(self.library.persons) != 1 else ''}.")

	# ---------- Liste des personnes ----------

	def refresh_persons(self, select: Optional[list[int]] = None) -> None:
		"""Reconstruit la liste des personnes : nommées d'abord (ordre alphabétique), puis inconnues (les plus grandes d'abord).

		Args:
			select (Optional[list[int]]): Identifiants des personnes à sélectionner ensuite ; `None` pour conserver la sélection actuelle. Faute de personne à sélectionner, la première de la liste l'est.
		"""
		if select is None:
			select = [int(i) for i in self.tree.selection()]
		self.tree.delete(*self.tree.get_children())
		if len(self.icon_cache) > ICON_CACHE_SIZE:
			self.icon_cache.clear()                  # Évite une croissance sans limite de la mémoire
		persons = sorted(
			self.library.persons.values(),
			key = lambda p: (p.name is None, (p.name or "").casefold(), -len(p.faces))
		)
		# Une icône par personne : son visage le plus net. Les vignettes manquantes sont lues par lots.
		bests = {person.identifier: max(person.faces, key = lambda f: f.score) for person in persons}
		thumbnails = self.database.thumbnails([face.face_id for face in bests.values() if face.face_id not in self.icon_cache])
		for person in persons:
			best = bests[person.identifier]
			icon = self.icon_cache.get(best.face_id)
			if icon is None:
				icon = thumbnail_photo(thumbnails.get(best.face_id), TREE_ROW_HEIGHT - 4)
				self.icon_cache[best.face_id] = icon
			label = emoji(("🙈 " if person.ignored else "") + (person.name or f"❓ Inconnu n°{person.identifier}"))
			origin = {
				"tag": "🏷️ étiquette",
				"manual": "✍️ saisi"}.get(person.source,
				""
			)
			self.tree.insert(
				"",
				"end",
				iid = str(person.identifier),
				text = " " + label,
				image = icon,
				values = (len(person.faces), len(person.photos), emoji(origin), emoji(self.person_status(person))),
			)
		still_there = [str(i) for i in select if i in self.library.persons]
		if not still_there and persons:
			still_there = [str(persons[0].identifier)]   # Par défaut, on affiche la première personne
		if still_there:
			self.tree.selection_set(still_there)
		else:
			self.show_person(None)
		self.update_action_states()

	def write_fields(self) -> tuple[bool, bool, bool]:
		"""Lit les cases « Écrire ».

		Returns:
			tuple[bool, bool, bool]: États des cases des régions de visages, de `PersonInImage` et des mots-clés.
		"""
		return self.write_regions_var.get(), self.write_person_var.get(), self.write_keywords_var.get()

	def missing_photos(self, person: Person) -> list[str]:
		"""Liste les photos de la personne où son nom manque encore, pour les cases « Écrire » cochées.

		Args:
			person (Person): Personne concernée.

		Returns:
			list[str]: Chemins triés des photos à écrire (voir `photos_missing_name`).
		"""
		return photos_missing_name(self.library, person, *self.write_fields())

	def person_status(self, person: Person) -> str:
		"""Calcule le texte de la colonne « Métadonnées » d'une personne.

		Args:
			person (Person): Personne de la ligne.

		Returns:
			str: « 🙈 ignorée », « ✅ à jour » si son nom est déjà dans toutes ses photos, ou le nombre de photos à écrire ; vide quand il n'y a rien à évaluer (personne sans nom, ExifTool absent, aucune case « Écrire » cochée).
		"""
		if person.ignored:
			return "🙈 ignorée"
		if not person.name or not self.exiftool.available or not any(self.write_fields()):
			return ""
		missing = len(self.missing_photos(person))
		# ✏ sans le sélecteur de variante U+FE0F, que Tk dessine comme un large blanc
		return "✅ à jour" if missing == 0 else f"✏ {missing} à écrire"

	def update_person_statuses(self) -> None:
		"""Met à jour la seule colonne « Métadonnées » (après une écriture, ou quand une case « Écrire » change), sans reconstruire la liste : la sélection, la page et la position dans la page sont conservées."""
		for iid in self.tree.get_children():
			person = self.library.persons.get(int(iid))
			if person is not None:
				self.tree.set(iid, "status", emoji(self.person_status(person)))

	def update_action_states(self) -> None:
		"""« Valider la personne » n'est actif que pour une seule personne, nommée et non ignorée, hors travail en cours."""
		selected = self.selected_persons()
		ready = not self.busy and len(selected) == 1 and bool(selected[0].name) and not selected[0].ignored
		self.validate_button.configure(state = "normal" if ready else "disabled")

	def tree_tooltip(self, event: tk.Event) -> Optional[str]:
		"""Calcule l'infobulle de la liste des personnes à l'endroit survolé.

		Sur les en-têtes, elle décrit le rôle de la colonne ; sur les lignes, elle résume la personne (utile quand son nom est trop long pour la colonne).

		Args:
			event (tk.Event): Événement de la souris sur la liste.

		Returns:
			Optional[str]: Texte de l'infobulle, ou None hors des en-têtes et des lignes.
		"""
		region = self.tree.identify_region(event.x, event.y)
		if region == "heading":
			return self.heading_tips.get(self.tree.identify_column(event.x))
		row = self.tree.identify_row(event.y) if region in ("tree", "cell") else ""
		person = self.library.persons.get(int(row)) if row else None
		if person is None:
			return None
		if person.ignored:
			status = "🙈 Ignorée : ni proposée par « Nommer les inconnus », ni écrite dans les métadonnées."
		elif not person.name:
			status = "❓ Pas encore de nom : double-clic pour la nommer."
		elif person.source == "tag":
			status = "🏷️ Nom lu dans les métadonnées des photos."
		else:
			status = "✍️ Nom saisi dans Face Tagger."
		if person.name and not person.ignored and self.exiftool.available and any(self.write_fields()):
			missing = len(self.missing_photos(person))
			if missing:
				status += (f"\n✏️ Nom absent de {missing} photo{'s' if missing != 1 else ''} : "
					"« Valider la personne » l'y écrit.")
			else:
				status += "\n✅ Nom présent dans toutes ses photos : rien à écrire."
		return (
			f"👤 {person.display_name} — {len(person.faces)} visage{'s' if len(person.faces) != 1 else ''} dans {len(person.photos)} photo{'s' if len(person.photos) != 1 else ''}\n{status}\n"
			"🖱️ Double-clic : renommer — Ctrl ou Maj + clic : sélectionner plusieurs personnes"
		)

	def selected_persons(self) -> list[Person]:
		"""Liste les personnes sélectionnées dans la liste.

		Returns:
			list[Person]: Personnes sélectionnées qui existent encore dans la photothèque.
		"""
		return [self.library.persons[int(i)] for i in self.tree.selection() if int(i) in self.library.persons]

	def on_select(self, event: Optional[tk.Event] = None) -> None:
		"""Réagit à un changement de sélection dans la liste des personnes.

		Affiche la personne choisie (ou le nombre de personnes sélectionnées) et met à jour les boutons.

		Args:
			event (Optional[tk.Event]): Événement `<<TreeviewSelect>>` (inutilisé), ou `None` lors d'un appel direct.
		"""
		selected = self.selected_persons()
		# La page courante est conservée tant que la personne affichée ne change pas
		self.show_person(selected[0] if len(selected) == 1 else None, keep_page = True)
		if len(selected) > 1:
			self.person_title.set_text(f"👥 {len(selected)} personnes sélectionnées")
		self.update_action_states()

	def page_size(self) -> int:
		"""Lit le nombre de vignettes par page choisi dans l'interface.

		Returns:
			int: Nombre de vignettes par page : au moins 10, et 100 si la valeur saisie est illisible.
		"""
		try:
			return max(10, int(self.page_size_var.get()))
		except (ValueError, tk.TclError):
			return 100

	def show_person(self, person: Optional[Person], keep_page: bool = False) -> None:
		"""Affiche les visages d'une personne, triés du plus net au moins net puis découpés en pages.

		La sélection de vignettes est vidée si la personne affichée change.

		Args:
			person (Optional[Person]): Personne à afficher, ou `None` pour vider la zone des vignettes.
			keep_page (bool): `True` pour rester sur la page courante si la personne affichée ne change pas.
		"""
		changed = person is not self.current_person
		self.current_person = person
		self.sorted_faces = sorted(person.faces, key = lambda f: -f.score) if person is not None else []
		self.face_index = {face: index for index, face in enumerate(self.sorted_faces)}
		if changed:
			# La sélection de vignettes ne vaut que pour la personne affichée
			self.selected_faces = set()
			self.selection_anchor = None
		else:
			# Visages partis entre-temps (déplacés, détachés, écartés) : ils quittent la sélection
			self.selected_faces &= set(self.sorted_faces)
			if self.selection_anchor is not None and self.selection_anchor >= len(self.sorted_faces):
				self.selection_anchor = None
		if changed or not keep_page:
			self.page = 0
		self.render_page()

	def go_to_page(self, page: int) -> None:
		"""Affiche une page des vignettes de la personne.

		Args:
			page (int): Rang de la page, à partir de `0` ; il est ramené dans les limites existantes.
		"""
		self.page = page
		self.render_page()

	def render_page(self) -> None:
		"""Construit les vignettes de la page courante. Les images des autres pages sont libérées : la mémoire utilisée dépend de la taille de la page, pas du nombre de visages de la personne."""
		for child in self.faces_frame.winfo_children():
			child.destroy()
		self.face_images.clear()
		self.face_labels.clear()
		self.face_widgets.clear()
		person = self.current_person
		if person is None:
			self.person_title.set_text("")
			self.page_label.configure(text = "Page – / –")
			self.range_label.configure(text = "")
			for button in (self.first_button, self.previous_button, self.next_button, self.last_button):
				button.configure(state = "disabled")
			self.update_face_selection()
			return

		size = self.page_size()
		total = len(self.sorted_faces)
		pages = max(1, math.ceil(total / size))
		self.page = min(max(self.page, 0), pages - 1)
		chunk = self.sorted_faces[self.page * size:(self.page + 1) * size]

		self.update_person_title()
		self.page_label.configure(text = f"Page {self.page + 1} / {pages}")
		self.range_label.configure(text = f"{self.page * size + 1 if chunk else 0}–{self.page * size + len(chunk)} sur {total}")
		for button in (self.first_button, self.previous_button):
			button.configure(state = "normal" if self.page > 0 else "disabled")
		for button in (self.next_button, self.last_button):
			button.configure(state = "normal" if self.page < pages - 1 else "disabled")

		# Les vignettes de la page sont lues dans la base en une seule requête
		thumbnails = self.database.thumbnails([face.face_id for face in chunk])
		for face in chunk:
			# Cadre vert : visage déjà marqué dans la photo ; cadre bleu : visage proposé, nom pas encore écrit ;
			# cadre rouge : photo en erreur (extension incorrecte…), nom impossible à écrire
			image = thumbnail_photo(thumbnails.get(face.face_id), border = self.face_frame_color(face))
			self.face_images.append(image)
			label = ttk.Label(
				self.faces_frame,
				image = image,
				text = os.path.basename(face.photo)[:16],
				compound = "top",
				style = "Face.TLabel"
			)
			# Sélection comme dans l'Explorateur : clic simple, Ctrl + clic (Commande + clic sous macOS), Maj + clic
			label.bind("<Button-1>", lambda event, f = face: self.click_face(f, "single"))
			label.bind("<Control-Button-1>", lambda event, f = face: self.click_face(f, "toggle"))
			label.bind("<Shift-Button-1>", lambda event, f = face: self.click_face(f, "range"))
			if sys.platform == "darwin":
				label.bind("<Command-Button-1>", lambda event, f = face: self.click_face(f, "toggle"))
			label.bind("<Double-Button-1>", lambda event, f = face: self.open_photo(f))
			label.bind("<Button-3>", lambda event, f = face: self.face_menu(event, f))
			label.bind("<Button-2>", lambda event, f = face: self.face_menu(event, f))   # Clic droit sous macOS
			label.bind("<MouseWheel>", lambda event: self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"))
			label.bind("<Button-4>", lambda event: self.canvas.yview_scroll(-1, "units"))
			label.bind("<Button-5>", lambda event: self.canvas.yview_scroll(1, "units"))
			# Après les liaisons ci-dessus, que l'infobulle complète sans les remplacer ; son texte est calculé à
			# l'affichage, car l'état du visage (marqué ou proposé) change après une écriture
			ToolTip(label, lambda event, f = face: self.face_tooltip(f))
			self.face_labels.append(label)
			self.face_widgets[face] = label
		self.update_face_selection()
		self.layout_faces(force = True)
		self.canvas.yview_moveto(0)

	def update_person_title(self) -> None:
		"""Écrit le titre de la personne affichée : nom, nombre de visages déjà marqués, proposés et en erreur, nombre de photos."""
		person = self.current_person
		if person is None:
			self.person_title.set_text("")
			return
		total = len(person.faces)
		# Les visages des photos en erreur sont comptés à part : leur cadre est rouge, quel que soit leur état
		errors = sum(1 for face in person.faces if face.photo in self.library.photo_errors)
		tagged = sum(1 for face in person.faces if face.photo not in self.library.photo_errors and self.face_tagged(face))
		proposed = total - tagged - errors
		photos = len(person.photos)
		error_text = f", 🟥 {errors} en erreur" if errors else ""
		self.person_title.set_text((
			f"{'🙈 ' if person.ignored else '👤 '}{person.display_name} — {total} visage{'s' if total != 1 else ''} "
			f"(🟩 {tagged} marqué{'s' if tagged > 1 else ''}, 🟦 {proposed} proposé{'s' if proposed > 1 else ''}{error_text}), "
			f"{photos} photo{'s' if photos != 1 else ''}"
		))

	def person_title_tooltip(self, event: tk.Event) -> Optional[str]:
		"""Calcule l'infobulle du titre de la personne affichée : la légende des cadres des vignettes.

		Args:
			event (tk.Event): Événement de la souris (inutilisé).

		Returns:
			Optional[str]: Légende des couleurs, ou `None` quand aucune personne n'est affichée (titre vide, ou plusieurs personnes sélectionnées).
		"""
		if self.current_person is None:
			return None
		return ("🟩 Cadre vert : visage déjà marqué, son nom figure dans une région de la photo.\n"
			"🟦 Cadre bleu : visage proposé, son nom n'est pas encore écrit dans la photo.\n"
			"✅ « Valider la personne » écrit son nom : ses cadres bleus deviennent verts.\n"
			"🟥 Cadre rouge : photo en erreur (extension qui ne correspond pas au contenu…), son nom ne peut pas y être écrit ; double-clic pour le détail.")

	def face_tagged(self, face: FaceRecord) -> bool:
		"""Indique si un visage de la personne affichée est déjà marqué dans sa photo (voir `face_is_tagged`).

		Args:
			face (FaceRecord): Visage de la personne affichée.

		Returns:
			bool: `True` si une région de la photo porte déjà le nom de la personne à l'emplacement du visage.
		"""
		name = self.current_person.name if self.current_person is not None else None
		return face_is_tagged(face, name, self.library.photo_tags.get(face.photo))

	def face_frame_color(self, face: FaceRecord) -> str:
		"""Choisit la couleur du cadre d'une vignette.

		Args:
			face (FaceRecord): Visage de la personne affichée.

		Returns:
			str: Rouge (`FACE_FRAME_COLORS["error"]`) pour un visage d'une photo en erreur, sinon vert (`FACE_FRAME_COLORS["tagged"]`) pour un visage déjà marqué dans sa photo, bleu (`FACE_FRAME_COLORS["proposed"]`) pour un visage proposé.
		"""
		# L'erreur passe avant tout : tant qu'elle demeure, le nom ne peut pas être écrit dans la photo
		if face.photo in self.library.photo_errors:
			return FACE_FRAME_COLORS["error"]
		return FACE_FRAME_COLORS["tagged" if self.face_tagged(face) else "proposed"]

	def face_tooltip(self, face: FaceRecord) -> str:
		"""Calcule l'infobulle d'une vignette : photo, état du visage (marqué, proposé ou en erreur) et commandes de la souris.

		Args:
			face (FaceRecord): Visage de la vignette survolée.

		Returns:
			str: Texte de l'infobulle.
		"""
		error = self.library.photo_errors.get(face.photo)
		if error:
			state = f"🟥 Photo en erreur : {error}"
		elif self.face_tagged(face):
			state = "🟩 Déjà marqué : son nom figure dans la photo."
		else:
			state = "🟦 Proposé : son nom n'est pas encore écrit dans la photo."
		return (photo_caption(face.photo) + f"\n{state}\n🖱️ Clic : sélectionner — Ctrl ou Maj + clic : sélection multiple\n"
			"🖱️ Double-clic : ouvrir la photo — clic droit : déplacer, détacher ou écarter")

	def update_face_frames(self) -> None:
		"""Redessine le cadre des vignettes de la page et le titre, sans reconstruire la page.

		Appelée quand les étiquettes connues changent (après une écriture) : les visages dont le nom vient d'être écrit passent du bleu au vert, tandis que la sélection et la position dans la page sont conservées.
		"""
		if self.current_person is None or not self.face_widgets:
			return
		# Les vignettes de la page sont dans le cache de la base : les relire ne coûte presque rien
		thumbnails = self.database.thumbnails([face.face_id for face in self.face_widgets])
		images: list[ImageTk.PhotoImage] = []
		for face, label in self.face_widgets.items():
			image = thumbnail_photo(thumbnails.get(face.face_id), border = self.face_frame_color(face))
			images.append(image)
			label.configure(image = image)
		# Les anciennes images ne sont libérées qu'une fois remplacées dans les widgets
		self.face_images = images
		self.update_person_title()

	def layout_faces(self, force: bool = False) -> None:
		"""Place les vignettes de la page en grille, selon la largeur de la zone.

		Args:
			force (bool): `True` pour replacer les vignettes même si le nombre de colonnes n'a pas changé.
		"""
		columns = max(1, self.canvas.winfo_width() // scaled(THUMBNAIL_SIZE + 24))
		if columns == self.grid_columns and not force:
			return
		self.grid_columns = columns
		for position, label in enumerate(self.face_labels):
			label.grid(
				row = position // columns,
				column = position % columns,
				padx = 4,
				pady = 4
			)

	# ---------- Sélection de plusieurs vignettes ----------

	def click_face(self, face: FaceRecord, mode: str) -> None:
		"""Sélectionne des vignettes comme dans l'Explorateur.

		Args:
			face (FaceRecord): Visage de la vignette cliquée.
			mode (str): « `single` » (clic) : cette vignette seule ; « `toggle` » (Ctrl + clic) : ajoute la vignette à la sélection, ou l'en retire ; « `range` » (Maj + clic) : toutes les vignettes entre la dernière cliquée et celle-ci, pages précédentes comprises (les rangs sont ceux de la liste complète des visages).
		"""
		# Les raccourcis clavier (Ctrl + A, Échap, Suppr) sont liés au canevas : il prend le focus
		self.canvas.focus_set()
		index = self.face_index.get(face)
		if index is None:
			return
		if mode == "toggle":
			if face in self.selected_faces:
				self.selected_faces.discard(face)
			else:
				self.selected_faces.add(face)
			self.selection_anchor = index
		elif mode == "range" and self.selection_anchor is not None:
			low, high = sorted((self.selection_anchor, index))
			self.selected_faces = set(self.sorted_faces[low:high + 1])
		else:
			self.selected_faces = {face}
			self.selection_anchor = index
		self.update_face_selection()

	def select_all_faces(self) -> None:
		"""Sélectionne tous les visages de la personne affichée, toutes pages confondues."""
		self.selected_faces = set(self.sorted_faces)
		self.update_face_selection()

	def clear_face_selection(self) -> None:
		"""Désélectionne toutes les vignettes (touche Échap, ou clic dans le vide)."""
		self.canvas.focus_set()
		self.selected_faces = set()
		self.selection_anchor = None
		self.update_face_selection()

	def update_face_selection(self) -> None:
		"""Met en évidence les vignettes sélectionnées et active les commandes qui s'y appliquent."""
		for face, label in self.face_widgets.items():
			label.configure(style = "SelectedFace.TLabel" if face in self.selected_faces else "Face.TLabel")
		count = len(self.selected_faces)
		if count:
			text = f"☑ {count} visage{'s' if count != 1 else ''} sélectionné{'s' if count != 1 else ''}"
		else:
			text = "Aucun visage sélectionné"
		self.selection_label.configure(text = emoji(text))
		for button in self.selection_buttons:
			button.configure(state = "normal" if count else "disabled")
		self.select_all_button.configure(state = "normal" if self.sorted_faces else "disabled")

	def selected_face_list(self) -> list[FaceRecord]:
		"""Liste les visages sélectionnés, dans l'ordre d'affichage.

		Seuls comptent ceux qui appartiennent encore à la personne affichée : une autre action (fusion, renommage) a pu les en retirer depuis.

		Returns:
			list[FaceRecord]: Visages sélectionnés, du plus net au moins net.
		"""
		return [
			face for face in self.sorted_faces
			if face in self.selected_faces and face.person is self.current_person and not face.deleted
		]

	def after_faces_changed(
			self,
			origin: Optional[Person],
			fallback: Optional[Person] = None
		) -> None:
		"""Vide la sélection et reconstruit la liste après un déplacement ou une suppression de visages.

		Args:
			origin (Optional[Person]): Personne affichée avant l'opération, qui le reste si elle existe encore.
			fallback (Optional[Person]): Personne à afficher si `origin` n'existe plus ; à défaut, la première de la liste.
		"""
		self.selected_faces = set()
		self.selection_anchor = None
		if origin is not None and origin.identifier in self.library.persons:
			select: list[int] = [origin.identifier]
		elif fallback is not None and fallback.identifier in self.library.persons:
			select = [fallback.identifier]
		else:
			select = []
		self.refresh_persons(select = select)

	# ---------- Actions sur les personnes et les visages ----------

	def known_names(self) -> list[str]:
		"""Liste les noms déjà donnés, pour les boîtes de saisie.

		Returns:
			list[str]: Noms, triés sans tenir compte de la casse.
		"""
		return sorted({p.name for p in self.library.persons.values() if p.name}, key = str.casefold)

	def rename_selected(self) -> None:
		"""Renomme la personne sélectionnée ; un nom déjà utilisé réunit les deux groupes."""
		selected = self.selected_persons()
		if len(selected) != 1:
			return
		person = selected[0]
		dialog = NameDialog(
			self,
			"✏️ Renommer", "Nom de la personne (un nom existant fusionne les deux groupes) :",
			self.known_names(),
			initial = person.name or ""
		)
		if dialog.result:
			merged = self.library.rename(person, dialog.result)
			self.log(f"✍️ « {merged.name} » : {len(merged.faces)} visage{'s' if len(merged.faces) != 1 else ''}")
			self.refresh_persons(select = [merged.identifier])

	def merge_selected(self) -> None:
		"""Fusionne les personnes sélectionnées (confirmation si elles portent des noms différents)."""
		persons = self.selected_persons()
		if len(persons) < 2:
			messagebox.showinfo(
				emoji("🔗 Fusionner"),
				"Sélectionnez au moins deux personnes (Ctrl ou Maj + clic)."
			)
			return
		if len({name_key(p.name) for p in persons if p.name}) > 1 and not messagebox.askyesno(
			emoji("🔗 Fusionner"),
			"Ces personnes portent des noms différents. Fusionner quand même ?",
		):
			return
		merged = self.library.merge(persons)
		self.log(f"🔗 {len(persons)} groupes fusionnés en « {merged.display_name} »")
		self.refresh_persons(select = [merged.identifier])

	def toggle_ignore(self) -> None:
		"""Ignore les personnes sélectionnées, ou les réactive si elles l'étaient déjà."""
		for person in self.selected_persons():
			person.ignored = not person.ignored
		self.refresh_persons()
		self.show_person(self.current_person, keep_page = True)

	def name_unknowns(self) -> None:
		"""Ouvre la boîte qui présente une à une les personnes sans nom."""
		if not any(not p.name and not p.ignored for p in self.library.persons.values()):
			messagebox.showinfo(
				emoji("🎉 Rien à faire"),
				"Toutes les personnes sont déjà nommées ou ignorées."
			)
			return
		UnknownNamesDialog(self)
		self.refresh_persons()

	def open_photo(self, face: FaceRecord) -> None:
		"""Affiche la photo entière d'un visage, avec le cadre et le nom de chaque visage.

		Args:
			face (FaceRecord): Visage dont on ouvre la photo.
		"""
		PhotoViewer(self, self.library, face)

	def face_menu(self, event: tk.Event, face: FaceRecord) -> None:
		"""Ouvre le menu contextuel d'une vignette.

		Ses commandes portent sur toute la sélection si la vignette en fait partie, sinon sur cette seule vignette, qui devient alors la sélection.

		Args:
			event (tk.Event): Clic droit, qui donne la position du menu à l'écran.
			face (FaceRecord): Visage de la vignette cliquée.
		"""
		if face not in self.selected_faces:
			self.click_face(face, "single")
		count = len(self.selected_face_list())
		suffix = f" ({count} visages)" if count > 1 else ""
		menu = tk.Menu(self, tearoff = False)
		menu.add_command(
			label = emoji("🖼️ Ouvrir la photo"),
			command = lambda: self.open_photo(face)
		)
		menu.add_separator()
		menu.add_command(
			label = emoji(f"↪️ Déplacer vers une personne…{suffix}"),
			command = self.move_selected_faces
		)
		menu.add_command(
			label = emoji(f"✂️ Détacher dans un nouveau groupe{suffix}"),
			command = self.detach_selected_faces
		)
		menu.add_command(
			label = emoji(f"🗑️ Ce n'est pas un visage (écarter){suffix}"),
			command = self.delete_selected_faces
		)
		menu.tk_popup(event.x_root, event.y_root)

	def move_selected_faces(self) -> None:
		"""Déplace les visages sélectionnés vers une personne choisie par son nom (créée si elle n'existe pas)."""
		faces = self.selected_face_list()
		if not faces:
			return
		count = len(faces)
		dialog = NameDialog(
			self,
			"↪️ Déplacer les visages" if count > 1 else "↪️ Déplacer le visage",
			f"Nom de la personne de destination ({count} visage{'s' if count != 1 else ''}) :",
			self.known_names()
		)
		if not dialog.result:
			return
		target = self.library.person_by_name(dialog.result) or self.library.new_person(dialog.result, "manual")
		origin = self.current_person
		if target is origin:
			return
		for face in faces:
			self.library.move_face(face, target)
		self.log(f"↪️ {count} visage{'s' if count != 1 else ''} déplacé{'s' if count != 1 else ''} vers « {target.display_name} »")
		self.after_faces_changed(origin, fallback = target)

	def detach_selected_faces(self) -> None:
		"""Réunit les visages sélectionnés dans un nouveau groupe sans nom, qui est ensuite affiché."""
		faces = self.selected_face_list()
		if not faces:
			return
		person = self.library.new_person()
		for face in faces:
			self.library.move_face(face, person)
		self.log(f"✂️ {len(faces)} visage{'s' if len(faces) != 1 else ''} détaché{'s' if len(faces) != 1 else ''} dans « {person.display_name} »")
		self.after_faces_changed(person)

	def delete_selected_faces(self) -> None:
		"""Écarte les visages sélectionnés (faux positifs). Confirmation demandée au-delà d'un visage."""
		faces = self.selected_face_list()
		if not faces:
			return
		if len(faces) > 1 and not messagebox.askyesno(
			emoji("🗑️ Écarter"),
			f"Écarter {len(faces)} visages ? Ils ne seront plus affichés ni écrits dans les métadonnées "
			"(une nouvelle analyse les fait réapparaître).",
		):
			return
		origin = self.current_person
		for face in faces:
			self.library.delete_face(face)
		self.log(f"🗑️ {len(faces)} visage{'s' if len(faces) != 1 else ''} écarté{'s' if len(faces) != 1 else ''}")
		self.after_faces_changed(origin)

	# ---------- Écriture des métadonnées (thread de travail) ----------

	def collect_updates(
			self,
			persons: list[Person]
		) -> dict[str, list[tuple[str, FaceRecord]]]:
		"""Regroupe par photo les noms à écrire pour les personnes données.

		Les personnes sans nom ou ignorées sont sautées, et seules les photos où le nom manque encore sont retenues : les autres sont déjà à jour.

		Args:
			persons (list[Person]): Personnes dont on veut écrire le nom.

		Returns:
			dict[str, list[tuple[str, FaceRecord]]]: Couples (nom, visage) à écrire, par chemin de photo.
		"""
		updates: dict[str, list[tuple[str, FaceRecord]]] = {}
		for person in persons:
			if not person.name or person.ignored:
				continue
			missing = set(self.missing_photos(person))
			for face in person.faces:
				if face.photo in missing:
					updates.setdefault(face.photo, []).append((person.name, face))
		return updates

	def can_write(self) -> bool:
		"""Vérifie qu'une écriture est possible, et sinon l'explique dans une boîte de dialogue.

		Returns:
			bool: `True` si ExifTool est présent et qu'au moins une case « Écrire » est cochée.
		"""
		if not self.exiftool.available:
			messagebox.showerror(
				emoji("❌ ExifTool manquant"),
				"Installez ExifTool pour écrire les métadonnées."
			)
			return False
		if not any(self.write_fields()):
			messagebox.showinfo(
				emoji("💾 Écriture"),
				"Cochez au moins un type de métadonnées à écrire."
			)
			return False
		return True

	def backup_note(self) -> str:
		"""Rédige le rappel de la sauvegarde *_original, ajouté aux demandes de confirmation.

		Returns:
			str: Phrase précédée d'un saut de ligne, qui annonce la copie *_original ou son absence.
		"""
		if self.backup_var.get():
			return "\nUne copie *_original de chaque photo sera conservée."
		return "\n⚠️ Les photos seront modifiées sans copie de sauvegarde."

	def validate_person(self) -> None:
		"""Écrit dans les métadonnées le nom d'une seule personne, celle qui est sélectionnée, après vérification de ses vignettes par l'utilisateur."""
		selected = self.selected_persons()
		if len(selected) != 1:
			messagebox.showinfo(emoji("✅ Valider"), "Sélectionnez une seule personne.")
			return
		person = selected[0]
		if not person.name:
			messagebox.showinfo(emoji("✅ Valider"), "Nommez d'abord cette personne (double-clic dans la liste).")
			return
		if person.ignored:
			messagebox.showinfo(emoji("✅ Valider"), "Cette personne est ignorée : réactivez-la d'abord.")
			return
		if not self.can_write():
			return
		updates = self.collect_updates([person])
		if not updates:
			messagebox.showinfo(
				emoji("✅ Valider"),
				emoji(f"✅ « {person.name} » figure déjà dans les métadonnées de ses {len(person.photos)} photo{'s' if len(person.photos) != 1 else ''}."),
			)
			return
		count = len(updates)
		if not messagebox.askyesno(
			emoji("✅ Valider"),
			emoji(f"Écrire « {person.name} » dans les métadonnées de {count} photo{'s' if count != 1 else ''} ?{self.backup_note()}"),
		):
			return
		self.launch_write(updates)

	def start_write(self) -> None:
		"""Écrit d'un coup le nom de toutes les personnes nommées et non ignorées."""
		if not self.can_write():
			return
		updates = self.collect_updates(list(self.library.persons.values()))
		if not updates:
			messagebox.showinfo(
				emoji("💾 Écriture"),
				"Rien à écrire : aucune personne nommée, ou leurs noms figurent déjà dans toutes leurs photos."
			)
			return
		names = {name for items in updates.values() for name, _ in items}
		if not messagebox.askyesno(
			emoji("💾 Écriture"),
			emoji(f"Écrire {len(names)} nom{'s' if len(names) != 1 else ''} dans les métadonnées de {len(updates)} photo{'s' if len(updates) != 1 else ''} ?{self.backup_note()}"),
		):
			return
		self.launch_write(updates)

	def launch_write(self, updates: dict[str, list[tuple[str, FaceRecord]]]) -> None:
		"""Lance l'écriture des métadonnées dans un thread de travail.

		Les cases sont lues ici, dans le thread de l'interface : le thread de travail ne touche jamais Tk.

		Args:
			updates (dict[str, list[tuple[str, FaceRecord]]]): Couples (nom, visage) à écrire, par chemin de photo (voir `collect_updates`).
		"""
		self.cancel_event.clear()
		self.set_busy(True)
		write_options = (
			self.write_regions_var.get(),
			self.write_person_var.get(),
			self.write_keywords_var.get(),
			self.backup_var.get()
		)
		self.worker = threading.Thread(
			target = self.write_worker,
			args = (updates, write_options),
			daemon = True
		)
		self.worker.start()

	def write_worker(
			self,
			updates: dict[str, list[tuple[str, FaceRecord]]],
			write_options: tuple
		) -> None:
		"""Écrit les métadonnées par lots de `EXIFTOOL_BATCH` photos (exécuté hors du thread Tk).

		Les photos dont l'extension ne correspond pas au contenu sont écartées (ExifTool les refuserait) et signalées à l'interface (événement « `errors` », comme les refus d'ExifTool). Après chaque lot, les photos sont relues pour vérifier l'écriture, puis la base est mise à jour (signature, étiquettes, historique) et les étiquettes relues sont envoyées à l'interface (événement « `tags` »). Les erreurs sont écrites dans le journal, et l'événement « `done` » est envoyé dans tous les cas.

		Args:
			updates (dict[str, list[tuple[str, FaceRecord]]]): Couples (nom, visage) à écrire, par chemin de photo.
			write_options (tuple): Quatre booléens : écrire les régions de visages, écrire `PersonInImage`, écrire les mots-clés, conserver une sauvegarde `*_original`.
		"""
		write_regions, write_person, write_keywords, backup = write_options
		fields = [label for flag, label in ((write_regions, "regions"), (write_person, "PersonInImage"), (write_keywords, "keywords")) if flag]
		try:
			photos = list(updates)
			self.post(
				"log",
				f"💾 Écriture des métadonnées de {len(photos)} photo{'s' if len(photos) != 1 else ''}…"
			)
			problems = verified_count = backups_count = 0
			total = len(photos)
			# Format vérifié de nouveau juste avant l'écriture : ExifTool refuse d'écrire dans une photo dont l'extension
			# ne correspond pas au contenu, inutile de la lui confier. L'événement efface aussi les erreurs disparues
			# (`None`) : une photo qui avait échoué lors d'une écriture précédente est ainsi retentée.
			format_errors: dict[str, Optional[str]] = {photo: photo_format_error(photo) for photo in photos}
			self.post("errors", format_errors)
			for photo, error in format_errors.items():
				if error:
					problems += 1
					self.post("log", f"🟥 {os.path.basename(photo)} non modifiée : {error}")
			photos = [photo for photo in photos if not format_errors[photo]]
			for start in range(0, len(photos), EXIFTOOL_BATCH):
				if self.cancel_event.is_set():
					self.post("log", "⏹️ Écriture interrompue.")
					break
				batch = photos[start:start + EXIFTOOL_BATCH]
				self.post("progress", start, len(photos), f"💾 Écriture {start}/{len(photos)}…")

				# 1. Un enregistrement par photo (fusion avec les étiquettes déjà présentes)
				records: list[dict] = []
				for photo in batch:
					tags = self.library.photo_tags.get(photo, PhotoTags())
					names = merge_unique([name for name, _ in updates[photo]])
					record: dict = {"SourceFile": Path(photo).as_posix()}
					# Régions d'abord : elles indiquent quels anciens noms sont remplacés (personne renommée)
					replacements: dict[str, str] = {}
					if write_regions:
						record["XMP-mwg-rs:RegionInfo"], replacements = build_region_info(tags, updates[photo])
						for old, new in replacements.items():
							self.post(
								"log",
								f"✏️ {os.path.basename(photo)} : « {old} » remplacé par « {new} »"
							)
					# Les anciens noms remplacés sont retirés des listes (sauf s'ils sont aussi voulus ailleurs)
					removed = {name_key(old) for old in replacements} - {name_key(name) for name in names}

					def kept(existing: list[str]) -> list[str]:
						"""Retire d'une liste existante les anciens noms remplacés par un renommage.

						Args:
							existing (list[str]): Noms déjà présents dans le champ.

						Returns:
							list[str]: Noms conservés, dans leur ordre d'origine.
						"""
						return [name for name in existing if name_key(name) not in removed]

					if write_person:
						record["XMP-iptcExt:PersonInImage"] = merge_unique(kept(tags.person_in_image), names)
					if write_keywords:
						# XMP est en UTF-8 par norme. L'IPTC n'est volontairement pas modifié : son jeu de caractères
						# n'est pas toujours déclaré et une écriture risquerait de corrompre les accents existants.
						record["XMP-dc:Subject"] = merge_unique(kept(tags.subject), names)
						record["XPKeywords"] = ";".join(merge_unique(kept(tags.xp_keywords), names))
					records.append(record)

				# 2. Écriture par ExifTool (la 1re version de chaque photo est conservée dans *_original)
				# Les erreurs d'ExifTool se terminent par « - <photo> » (chemin tel qu'il lui a été transmis) : elles
				# permettent d'encadrer en rouge les vignettes de la photo refusée et d'en afficher la cause dans l'aperçu.
				by_key: dict[str, str] = {path_key(Path(photo).as_posix()): photo for photo in batch}
				write_errors: dict[str, Optional[str]] = {}
				for message in self.exiftool.write_tags(records, backup):
					if "updated" in message or "unchanged" in message:
						self.post("log", f"✅ {message}")
					else:
						problems += 1
						self.post("log", f"⚠️ {message}")
						if message.startswith("Error") and " - " in message:
							cause, source = message.rsplit(" - ", 1)
							photo = by_key.get(path_key(source.strip()))
							if photo is not None:
								write_errors[photo] = f"ExifTool a refusé l'écriture : {cause.removeprefix('Error:').strip()}."

				# 3. Relecture : vérification des noms, puis mise à jour de la base
				entries = self.exiftool.read_tags(batch)
				fresh: dict[str, dict] = {}
				new_tags: dict[str, PhotoTags] = {}
				for photo in batch:
					entry = entries.get(path_key(photo))
					if entry is None:
						problems += 1
						self.post("log", f"⚠️ Relecture impossible : {os.path.basename(photo)}")
						continue
					names = merge_unique([name for name, _ in updates[photo]])
					tags = parse_photo_tags(entry)
					if check_written(tags, names, write_regions, write_person, write_keywords):
						verified = True
						verified_count += 1
						write_errors[photo] = None      # Noms bien écrits : la photo n'est plus en erreur
					else:
						verified = False
						problems += 1
						self.post(
							"log",
							f"⚠️ Noms attendus absents après écriture : {os.path.basename(photo)} (conflit avec une étiquette existante ?)"
						)
					try:
						# La taille du fichier a changé mais pas ses pixels : on met la signature à jour
						# et les visages mémorisés restent valables.
						stat = os.stat(photo)
						self.database.update_signature(photo, stat.st_mtime_ns, stat.st_size)
					except OSError:
						pass
					backup_path = photo + restore_originals.BACKUP_SUFFIX
					has_backup = os.path.exists(backup_path)
					backups_count += int(has_backup)
					self.database.record_modification(
						photo,
						names,
						fields,
						verified,
						backup_path if has_backup else None
					)
					fresh[photo] = entry
					new_tags[photo] = tags
				self.database.store_tags(fresh)
				self.post("tags", new_tags)
				if write_errors:
					self.post("errors", write_errors)

			self.post(
				"progress",
				total,
				total,
				f"✅ Écriture terminée ({problems} avertissement{'s' if problems != 1 else ''})."
			)
			self.post(
				"log",
				f"🎉 Terminé : {verified_count}/{total} photo{'s' if verified_count != 1 else ''} vérifiée{'s' if verified_count != 1 else ''} après relecture, {problems} avertissement{'s' if problems != 1 else ''}."
			)
			if backups_count:
				self.post(
					"log",
					f"📦 {backups_count} sauvegarde{'s' if backups_count != 1 else ''} *_original disponible{'s' if backups_count != 1 else ''} — bouton « Restaurer les sauvegardes » ou restore_originals.py."
				)
		except Exception as error:
			self.post("log", f"❌ Erreur pendant l'écriture : {error}")
		finally:
			self.post("done")

	# ---------- Restauration des sauvegardes *_original ----------

	def start_restore(self) -> None:
		"""Restaure d'un coup toutes les photos d'un dossier à partir de leurs sauvegardes `*_original`."""
		directory = filedialog.askdirectory(
			initialdir = self.base_directory(),
			title = "Dossier contenant les fichiers *_original",
		)
		if not directory:
			return
		backups = restore_originals.find_backups(os.path.abspath(directory))
		if not backups:
			messagebox.showinfo(
				emoji("ℹ️ Restauration"),
				"Aucune sauvegarde *_original trouvée dans ce dossier."
			)
			return
		if not messagebox.askyesno(
			emoji("♻️ Restauration"),
			emoji(f"Remplacer {len(backups)} photo{'s' if len(backups) != 1 else ''} par leur version d'origine ?\n"
				"Les noms écrits dans leurs métadonnées seront perdus, et la liste des personnes sera vidée "
				"si elle contient ces photos (relancez ensuite « Analyser »)."),
		):
			return
		self.set_busy(True)
		self.worker = threading.Thread(
			target = self.restore_worker,
			args = (backups,),
			daemon = True
		)
		self.worker.start()

	def restore_worker(self, backups: list[tuple[str, str]]) -> None:
		"""Restaure les photos depuis leurs sauvegardes, puis met la base à jour (exécuté hors du thread Tk).

		Les photos restaurées sont envoyées à l'interface (événement « `restored` ») ; l'événement « `done` » est envoyé dans tous les cas.

		Args:
			backups (list[tuple[str, str]]): Couples (sauvegarde, photo) trouvés par `restore_originals.find_backups`.
		"""
		try:
			self.post(
				"log",
				f"♻️ Restauration de {len(backups)} photo{'s' if len(backups) != 1 else ''}…"
			)
			done, failed = restore_originals.restore_backups(
				backups,
				database = self.database,
				log = lambda message: self.post("log", message),
			)
			self.post(
				"log",
				f"🎉 Terminé : {done} photo{'s' if done != 1 else ''} restaurée{'s' if done != 1 else ''}, {failed} échec{'s' if failed != 1 else ''}."
			)
			self.post(
				"restored",
				{target for _, target in backups}
			)
		except Exception as error:
			self.post("log", f"❌ Erreur pendant la restauration : {error}")
		finally:
			self.post("done")

	def after_restore(self, targets: set[str]) -> None:
		"""Vide la photothèque si des photos restaurées en font partie.

		Les étiquettes en mémoire ne correspondent plus aux fichiers restaurés : on repart d'une liste vide, et la prochaine analyse relit leurs étiquettes.

		Args:
			targets (set[str]): Chemins des photos restaurées.
		"""
		if targets & {face.photo for face in self.library.faces}:
			self.library = Library()
			self.refresh_persons()
			self.log("🔄 Liste des personnes vidée : relancez « Analyser » (rapide : seules les étiquettes des photos restaurées sont relues).")


# ======================================================================================
# Ligne de commande
# ======================================================================================

def parse_arguments() -> argparse.Namespace:
	"""Lit les options de la ligne de commande.

	Les valeurs par défaut qui dépendent d'autres options (seuil du modèle, chemin de la base) sont complétées ensuite par `main()`.

	Returns:
		argparse.Namespace: Options lues : `directory`, `data_directory`, `database`, `page_size`, `exiftool`, `model`, `detector`, `threshold`, `max_image_size`, `min_face_size`, `detection_score`, `no_backup`, `no_emoji` et `theme`.
	"""
	parser = argparse.ArgumentParser(
		description = "👥 Regroupe les visages d'un dossier de photos par personne et écrit les noms dans les métadonnées."
	)
	parser.add_argument(
		"--directory",
		default = "",
		help = "📁 Dossier de photos à ouvrir au démarrage",
	)
	parser.add_argument(
		"--data-directory",
		default = str(Path.home() / ".face_tagger"),
		help = "🗄️ Dossier des modèles de DeepFace et de la base",
	)
	parser.add_argument(
		"--database",
		default = "",
		help = "🗄️ Fichier de la base SQLite (défaut : <data-directory>/face_tagger.sqlite)",
	)
	parser.add_argument(
		"--page-size",
		type = int,
		default = 100,
		help = "📄 Nombre de vignettes affichées par page (défaut 100)",
	)
	parser.add_argument(
		"--exiftool",
		default = "exiftool",
		help = "🔧 Chemin ou nom de l'exécutable ExifTool",
	)
	parser.add_argument(
		"--model",
		choices = list(RECOGNITION_MODELS),
		default = DEFAULT_MODEL,
		help = f"🧠 Modèle de reconnaissance de DeepFace (défaut {DEFAULT_MODEL})",
	)
	parser.add_argument(
		"--detector",
		choices = DETECTOR_BACKENDS,
		default = DEFAULT_DETECTOR,
		help = f"🔍 Détecteur de visages de DeepFace (défaut {DEFAULT_DETECTOR} ; yunet est environ dix fois plus rapide)",
	)
	parser.add_argument(
		"--threshold",
		type = float,
		default = None,
		help = f"🎚️ Seuil de similarité cosinus ({THRESHOLD_MIN:.2f} à {THRESHOLD_MAX:.2f} ; défaut : celui du modèle, {RECOGNITION_MODELS[DEFAULT_MODEL]:.2f} pour {DEFAULT_MODEL})",
	)
	parser.add_argument(
		"--max-image-size",
		type = int,
		default = 2000,
		help = "📐 Taille maximale (pixels) des photos pour la détection",
	)
	parser.add_argument(
		"--min-face-size",
		type = int,
		default = 40,
		help = "🙂 Taille minimale (pixels) d'un visage retenu",
	)
	parser.add_argument(
		"--detection-score",
		type = float,
		default = 0.80,
		help = "🎯 Score minimal de détection d'un visage (0 à 1)",
	)
	parser.add_argument(
		"--no-backup",
		action = "store_true",
		help = "⚠️ Ne pas conserver de copie *_original lors de l'écriture",
	)
	parser.add_argument(
		"--no-emoji",
		action = "store_true",
		help = "🚫 Désactive les emojis (si Tk plante à leur affichage)",
	)
	parser.add_argument(
		"--theme",
		choices = ("auto", "light", "dark"),
		default = "auto",
		help = "🌓 Thème de l'interface : auto (celui de Windows, défaut), light (clair) ou dark (sombre)",
	)
	return parser.parse_args()


def main() -> None:
	"""Point d'entrée : prépare la console, lit les options puis ouvre la fenêtre principale."""
	# Exécutable fenêtré (PyInstaller --noconsole) : sys.stdout et sys.stderr valent None. Les barres de progression des téléchargements de DeepFace (tqdm) y écriraient et lèveraient une exception : ils sont redirigés vers le néant. Doit précéder l'analyse des arguments, qui écrit ses erreurs sur sys.stderr.
	for name in ("stdout", "stderr"):
		if getattr(sys, name) is None:
			setattr(sys, name, open(os.devnull, "w", encoding = "utf-8"))
	# Évite les erreurs d'encodage des emojis dans les consoles Windows, y compris dans l'aide de --help
	for stream in (sys.stdout, sys.stderr):
		if hasattr(stream, "reconfigure"):
			stream.reconfigure(encoding = "utf-8", errors = "replace")
	options = parse_arguments()
	SETTINGS["emoji"] = not options.no_emoji
	# « auto » ne suit que Windows : sous macOS, le thème natif de Tk suit déjà le mode sombre du système
	SETTINGS["dark"] = options.theme == "dark" or (options.theme == "auto" and windows_prefers_dark())
	# Sans seuil explicite, celui qui convient au modèle choisi : leurs échelles de ressemblance diffèrent
	if options.threshold is None:
		options.threshold = RECOGNITION_MODELS[options.model]
	options.threshold = min(max(options.threshold, THRESHOLD_MIN), THRESHOLD_MAX)
	if not options.database:
		options.database = str(Path(options.data_directory).expanduser() / "face_tagger.sqlite")
	app = FaceTaggerApp(options)
	app.mainloop()


if __name__ == "__main__":
	main()
