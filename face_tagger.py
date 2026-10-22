#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_tagger.py — Regroupe les visages d'une photothèque par personne et inscrit
les noms dans les métadonnées des photos.

	1. Parcours récursif d'un dossier de photos.
	2. Détection des visages (YuNet) et calcul d'une empreinte par visage (SFace) :
		deux petits modèles ONNX exécutés par OpenCV (pas de dlib, pas de GPU requis).
	3. Regroupement des visages par personne (similarité cosinus).
	4. Les étiquettes déjà présentes dans les photos (régions de visages MWG / Microsoft,
		PersonInImage, éventuellement mots-clés) servent à nommer automatiquement les groupes.
	5. Les personnes non reconnues peuvent être nommées à la main dans une boîte de dialogue.
	6. Écriture des noms via ExifTool : XMP-iptcExt:PersonInImage, régions de visages
		XMP-mwg-rs, mots-clés XMP-dc:Subject et EXIF XPKeywords (visible dans l'Explorateur Windows),
		suivie d'une relecture pour vérifier que les noms ont bien été écrits.
	7. Une base SQLite (face_database.py) mémorise visages, vignettes, étiquettes lues et historique
		des modifications : une photo inchangée n'est jamais ré-analysée ni relue.
	8. Les visages d'une personne sont affichés page par page (seules les vignettes de la page
		courante sont chargées en mémoire).
	9. Les sauvegardes *_original d'ExifTool se restaurent d'un coup (bouton, ou restore_originals.py).

Fichiers à placer dans le même dossier : face_tagger.py, face_database.py, restore_originals.py

Dépendances :
	pip install opencv-python pillow numpy      (+ pillow-heif pour les HEIC, optionnel)
	ExifTool   (Fedora : sudo dnf install perl-Image-ExifTool)
	Tkinter    (Fedora : sudo dnf install python3-tkinter)

Exemple :
	python face_tagger.py --directory ~/Images --threshold 0.40
"""

from __future__ import annotations

import argparse
import ctypes
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
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Optional

try:
	import cv2
	import numpy as np
	from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk
except ImportError as error:
	sys.exit(f"❌ Dépendance manquante : {error}\n👉 Installez-la avec : pip install opencv-python pillow numpy")

# Prise en charge optionnelle des HEIC/HEIF (photos d'iPhone) si pillow-heif est installé
try:
	import pillow_heif
	pillow_heif.register_heif_opener()
	HEIF_AVAILABLE = True
except ImportError:
	HEIF_AVAILABLE = False

# Modules compagnons (même dossier) : base SQLite et restauration des sauvegardes
try:
	from face_database import Database
	import restore_originals
except ImportError as error:
	sys.exit(f"❌ Fichier manquant : {error}\n👉 Placez face_tagger.py, face_database.py et restore_originals.py dans le même dossier.")


# ======================================================================================
# Constantes
# ======================================================================================

# Extensions d'images analysées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
if HEIF_AVAILABLE:
	IMAGE_EXTENSIONS |= {".heic", ".heif"}

# Modèles ONNX du « zoo » OpenCV, téléchargés automatiquement au premier lancement
MODEL_URLS = {
	"face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
	"face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
# En dessous de cette taille, le fichier est un pointeur Git LFS ou un téléchargement tronqué
MODEL_MIN_SIZE = 100_000

THUMBNAIL_SIZE = 96          # Côté (en pixels) des vignettes de visages
TREE_ROW_HEIGHT = 44         # Hauteur des lignes de la liste des personnes
MAX_FACES_SHOWN = 150        # Nombre maximal de vignettes affichées pour une personne
ICON_CACHE_SIZE = 4000       # Nombre maximal d'icônes de la liste des personnes gardées en mémoire
REGION_MATCH_IOU = 0.15      # Recouvrement minimal entre un visage détecté et une région existante
EXIFTOOL_BATCH = 25          # Nombre de photos modifiées par appel à ExifTool

# Identifiant de l'application pour la barre des tâches de Windows (voir set_windows_app_id)
APP_USER_MODEL_ID = "ILPlais.FaceTagger"

# Réglages globaux modifiables par la ligne de commande
SETTINGS = {"emoji": True}

# Caractères considérés comme des emojis (utile pour l'option --no-emoji)
EMOJI_PATTERN = re.compile("[\U0001F000-\U0001FFFF\u2190-\u21FF\u23E9-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")

# Étiquettes ExifTool lues dans chaque photo
TAGS_TO_READ = [
	"-XMP-mwg-rs:RegionInfo",        # Régions de visages (norme MWG : Lightroom, digiKam, Picasa…)
	"-XMP-MP:RegionInfoMP",          # Régions de visages « Microsoft Photo »
	"-XMP-iptcExt:PersonInImage",    # Liste de personnes présentes (sans position)
	"-XMP-dc:Subject",               # Mots-clés XMP
	"-IPTC:Keywords",                # Mots-clés IPTC
	"-XPKeywords",                   # Mots-clés EXIF de Windows
]


# ======================================================================================
# Fonctions utilitaires
# ======================================================================================

def emoji(text: str) -> str:
	"""Retourne le texte tel quel, ou débarrassé de ses emojis avec --no-emoji."""
	if SETTINGS["emoji"]:
		return text
	return EMOJI_PATTERN.sub("", text).strip()

def resource_path(relative_path):
	"""Obtient le chemin absolu vers la ressource, fonctionne pour le dev et PyInstaller."""
	try:
		# PyInstaller crée un dossier temporaire et stocke le chemin dans _MEIPASS
		base_path = sys._MEIPASS
	except AttributeError:
		# En développement : dossier du script, et non le dossier courant, pour trouver l'icône même quand
		# l'application est lancée depuis un autre dossier (raccourci, autre terminal…)
		base_path = os.path.dirname(os.path.abspath(__file__))

	return os.path.join(base_path, relative_path)


def set_windows_app_id() -> None:
	"""Donne à l'application sa propre identité dans la barre des tâches de Windows.

	Lancé par « python face_tagger.py », le programme est vu par Windows comme python.exe : la barre des
	tâches regroupe alors sa fenêtre avec Python et affiche l'icône de Python au lieu de celle de la fenêtre.
	Un identifiant d'application explicite (AppUserModelID) l'en sépare. L'identifiant vaut pour tout le
	processus et doit être déclaré avant la création de la première fenêtre.
	"""
	if sys.platform != "win32":
		return
	try:
		ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
	except (AttributeError, OSError):
		# Fonction absente ou appel refusé : seule l'icône de la barre des tâches est en jeu, on continue
		pass


def strip_widget_emoji(widget: tk.Misc) -> None:
	"""Retire récursivement les emojis des textes des widgets (option --no-emoji)."""
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
	"""Supprime les espaces superflus : « Jean   Dupont » devient « Jean Dupont »."""
	return " ".join(str(name).split())


def name_key(name: str) -> str:
	"""Clé de comparaison de deux noms, insensible à la casse et aux espaces."""
	return normalize_name(name).casefold()


def merge_unique(*lists) -> list[str]:
	"""Fusionne des listes de noms sans doublon (insensible à la casse), en gardant l'ordre."""
	seen: set[str] = set()
	result: list[str] = []
	for items in lists:
		for item in items:
			item = normalize_name(item)
			if item and name_key(item) not in seen:
				seen.add(name_key(item))
				result.append(item)
	return result


def as_list(value) -> list[str]:
	"""Convertit une valeur renvoyée par ExifTool (rien, texte, nombre ou liste) en liste de textes."""
	if value is None:
		return []
	if isinstance(value, list):
		return [str(item) for item in value]
	return [str(value)]


def path_key(path: str) -> str:
	"""Clé normalisée d'un chemin, pour faire correspondre les chemins de Python et d'ExifTool."""
	return os.path.normcase(os.path.normpath(path))


def iou(box_a: tuple, box_b: tuple) -> float:
	"""Intersection sur union de deux rectangles (x, y, largeur, hauteur)."""
	ax, ay, aw, ah = box_a
	bx, by, bw, bh = box_b
	inter_w = min(ax + aw, bx + bw) - max(ax, bx)
	inter_h = min(ay + ah, by + bh) - max(ay, by)
	if inter_w <= 0 or inter_h <= 0:
		return 0.0
	intersection = inter_w * inter_h
	return intersection / (aw * ah + bw * bh - intersection)


def list_photos(directory: str) -> list[str]:
	"""Liste récursivement les photos d'un dossier (dossiers et fichiers cachés ignorés)."""
	photos: list[str] = []
	for root, directories, files in os.walk(directory):
		# Modifier la liste « directories » sur place permet d'élaguer le parcours
		directories[:] = sorted(d for d in directories if not d.startswith("."))
		for file_name in sorted(files):
			if file_name.startswith("."):
				continue
			if Path(file_name).suffix.lower() in IMAGE_EXTENSIONS:
				photos.append(os.path.join(root, file_name))
	return photos


def load_font(size: int):
	"""Charge une police lisible pour annoter les photos (police par défaut en dernier recours)."""
	for font_name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
		try:
			return ImageFont.truetype(font_name, size)
		except OSError:
			continue
	return ImageFont.load_default()


def make_thumbnail(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
	"""Découpe une vignette carrée autour d'un visage (avec une marge) pour l'affichage."""
	x0, y0, x1, y1 = box
	center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
	half = max(x1 - x0, y1 - y0) * 0.75
	crop = image.crop((int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)))
	return ImageOps.fit(crop, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)


def encode_thumbnail(image: Image.Image) -> bytes:
	"""Encode une vignette en JPEG (stockée dans la base SQLite)."""
	buffer = io.BytesIO()
	image.save(buffer, format = "JPEG", quality = 85)
	return buffer.getvalue()


def decode_thumbnail(data: Optional[bytes]) -> Image.Image:
	"""Décode une vignette JPEG de la base (carré gris si elle est absente)."""
	if not data:
		return Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), "#888888")
	return Image.open(io.BytesIO(data)).convert("RGB")


def ensure_models(models_directory: Path, log: Callable[[str], None]) -> tuple[Path, Path]:
	"""Vérifie la présence des deux modèles ONNX et les télécharge si besoin."""
	models_directory.mkdir(parents = True, exist_ok = True)
	paths: list[Path] = []
	for file_name, url in MODEL_URLS.items():
		target = models_directory / file_name
		if not target.exists() or target.stat().st_size < MODEL_MIN_SIZE:
			log(f"⬇️ Téléchargement du modèle {file_name}…")
			partial = target.with_suffix(".part")
			try:
				urllib.request.urlretrieve(url, partial)
				if partial.stat().st_size < MODEL_MIN_SIZE:
					raise RuntimeError("fichier reçu trop petit (pointeur Git LFS ?)")
				partial.replace(target)
			except Exception as error:
				partial.unlink(missing_ok = True)
				raise RuntimeError(
					f"Téléchargement impossible de {file_name} ({error}).\n"
					f"👉 Téléchargez-le manuellement depuis {url}\n"
					f"   puis placez-le dans {models_directory}"
				) from error
		paths.append(target)
	return paths[0], paths[1]


# ======================================================================================
# ExifTool : lecture et écriture des métadonnées
# ======================================================================================

@dataclass
class PhotoTags:
	"""Étiquettes déjà présentes dans une photo."""
	regions: list[dict] = field(default_factory = list)          # [{"name", "box"}], box = (x, y, l, h) normalisé
	region_info: Optional[dict] = None                           # Structure MWG brute, conservée à l'écriture
	person_in_image: list[str] = field(default_factory = list)
	subject: list[str] = field(default_factory = list)
	keywords: list[str] = field(default_factory = list)
	xp_keywords: list[str] = field(default_factory = list)
	raw: dict = field(default_factory = dict)                    # Entrée JSON brute d'ExifTool (mémorisée en base)

	def weak_names(self, use_keywords: bool) -> list[str]:
		"""Noms « candidats » sans position dans l'image (PersonInImage et, au choix, mots-clés)."""
		names = list(self.person_in_image)
		if use_keywords:
			names += self.subject + self.keywords + self.xp_keywords
		return merge_unique(names)


def parse_photo_tags(entry: dict) -> PhotoTags:
	"""Transforme l'entrée JSON d'ExifTool pour une photo en objet PhotoTags."""
	tags = PhotoTags(raw = {key: value for key, value in entry.items() if key != "SourceFile"})

	# Régions MWG : X/Y désignent le CENTRE du rectangle, en coordonnées normalisées
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

	# Régions Microsoft : « Rectangle » = "x, y, largeur, hauteur" (coin supérieur gauche, normalisé)
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
	"""Petit adaptateur autour de l'exécutable ExifTool."""

	def __init__(self, executable: str = "exiftool"):
		self.path = shutil.which(executable)

	@property
	def available(self) -> bool:
		return self.path is not None

	def run(self, options: list[str], files: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
		"""Lance ExifTool. Les arguments (noms de fichiers compris) passent en UTF-8 par l'entrée standard
		(-@ -) : cela évite la limite de longueur de la ligne de commande et les soucis d'encodage des
		noms accentués, notamment sous Windows. `cwd` fixe le dossier de travail du processus."""
		arguments = ["-charset", "filename=utf8", *options, *files]
		return subprocess.run(
			[self.path, "-@", "-"],
			input = "\n".join(arguments).encode("utf-8"),
			capture_output = True,
			cwd = cwd,
			# Sous Windows, évite le clignotement d'une fenêtre console à chaque appel (lancement via pythonw)
			creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
		)

	def read_tags(self, photos: list[str], progress: Optional[Callable[[int, int], None]] = None) -> dict[str, dict]:
		"""Lit les étiquettes de plusieurs photos. Retourne {path_key(photo): entrée JSON}.
		Les photos qu'ExifTool n'a pas pu lire sont absentes du résultat."""
		result: dict[str, dict] = {}
		for start in range(0, len(photos), 100):
			chunk = photos[start:start + 100]
			completed = self.run(["-json", "-struct", *TAGS_TO_READ], [Path(p).as_posix() for p in chunk])
			try:
				entries = json.loads(completed.stdout.decode("utf-8", errors = "replace") or "[]")
			except json.JSONDecodeError:
				entries = []
			for entry in entries:
				result[path_key(entry.get("SourceFile", ""))] = entry
			if progress is not None:
				progress(min(start + len(chunk), len(photos)), len(photos))
		return result

	def write_tags(self, records: list[dict], backup: bool) -> list[str]:
		"""Écrit les métadonnées décrites par `records` (un dictionnaire par photo, avec SourceFile).
		Retourne les messages d'ExifTool (avertissements et erreurs compris)."""
		# Les enregistrements sont transmis dans un fichier JSON temporaire : c'est le moyen le plus fiable
		# d'écrire des structures XMP (régions) contenant des noms avec des caractères spéciaux.
		# ExifTool est lancé DANS le dossier temporaire et lit le JSON par un nom relatif : le chemin du
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
# Détection et reconnaissance des visages (OpenCV : YuNet + SFace)
# ======================================================================================

class FaceEngine:
	"""Détecte les visages d'une photo et calcule l'empreinte numérique de chacun."""

	def __init__(self, detection_model: Path, recognition_model: Path, max_image_size: int,
				min_face_size: int, detection_score: float):
		self.max_image_size = max_image_size
		self.min_face_size = min_face_size
		# Arguments positionnels : modèle, configuration, taille d'entrée, seuil de score, seuil NMS, top_k
		self.detector = cv2.FaceDetectorYN.create(str(detection_model), "", (320, 320), detection_score, 0.3, 5000)
		self.recognizer = cv2.FaceRecognizerSF.create(str(recognition_model), "")

	def analyse(self, photo: str) -> list[dict]:
		"""Retourne, pour chaque visage : boîte normalisée, score, empreinte (octets) et vignette JPEG (octets)."""
		with Image.open(photo) as source:
			# Pour les JPEG, le décodage à échelle réduite accélère beaucoup les gros fichiers
			source.draft("RGB", (self.max_image_size, self.max_image_size))
			# On redresse l'image selon l'orientation EXIF : les coordonnées sont celles de l'image affichée
			image = ImageOps.exif_transpose(source).convert("RGB")

		# Réduction éventuelle : la détection n'a pas besoin de 24 millions de pixels
		scale = self.max_image_size / max(image.size)
		if scale < 1:
			image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)

		bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
		height, width = bgr.shape[:2]
		self.detector.setInputSize((width, height))
		_, detections = self.detector.detect(bgr)

		results: list[dict] = []
		if detections is None:
			return results
		for row in detections:
			x, y, w, h = (float(value) for value in row[:4])
			# Rognage aux limites de l'image
			x0, y0 = max(0.0, x), max(0.0, y)
			x1, y1 = min(float(width), x + w), min(float(height), y + h)
			if min(x1 - x0, y1 - y0) < self.min_face_size:
				continue  # Visage trop petit : l'empreinte serait peu fiable
			# Alignement du visage (yeux, nez, bouche) puis calcul de l'empreinte à 128 dimensions
			aligned = self.recognizer.alignCrop(bgr, row)
			embedding = self.recognizer.feature(aligned).flatten().astype(np.float32)
			results.append({
				"box": (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height),
				"score": float(row[-1]),
				"embedding": embedding.tobytes(),                 # 128 flottants de 32 bits
				"thumbnail": encode_thumbnail(make_thumbnail(image, (x0, y0, x1, y1))),
			})
		return results


# ======================================================================================
# Modèle de données : visages, personnes, regroupement
# ======================================================================================

@dataclass(eq = False)
class FaceRecord:
	"""Un visage détecté dans une photo."""
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
	"""Un groupe de visages censés appartenir à la même personne."""
	identifier: int
	faces: list[FaceRecord] = field(default_factory = list)
	name: Optional[str] = None
	source: str = ""                             # "tag" (étiquette existante) ou "manual" (saisie)
	ignored: bool = False                        # Personne exclue de l'écriture des métadonnées

	@property
	def photos(self) -> set[str]:
		return {face.photo for face in self.faces}

	@property
	def display_name(self) -> str:
		return self.name or f"Inconnu n°{self.identifier}"


def cluster_faces(embeddings: np.ndarray, label_keys: list[Optional[str]], scores: list[float],
				threshold: float) -> list[list[int]]:
	"""Regroupe des empreintes de visages. Retourne une liste de groupes (listes d'indices).

	- Les visages déjà étiquetés sont d'abord regroupés par nom : l'étiquette fait foi.
	- Les autres rejoignent le groupe le plus proche si la similarité cosinus atteint le seuil,
		sinon ils fondent un nouveau groupe.
	- Les groupes devenus trop proches sont fusionnés, sauf s'ils portent des noms différents.
	"""
	count = len(embeddings)
	if count == 0:
		return []
	# Normalisation L2 : le produit scalaire de deux vecteurs devient leur similarité cosinus
	vectors = embeddings / np.maximum(np.linalg.norm(embeddings, axis = 1, keepdims = True), 1e-9)
	dimension = vectors.shape[1]

	members: list[list[int]] = []
	sums: list[np.ndarray] = []
	cluster_labels: list[Optional[str]] = []
	label_to_cluster: dict[str, int] = {}
	centroids = np.zeros((count, dimension), dtype = np.float32)

	def new_cluster(label: Optional[str]) -> int:
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

	# Passes de fusion : on réunit les groupes dont les centres sont proches
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
	"""Ensemble des visages, des personnes et des étiquettes existantes."""

	def __init__(self):
		self.faces: list[FaceRecord] = []
		self.persons: dict[int, Person] = {}
		self.photo_tags: dict[str, PhotoTags] = {}     # Étiquettes existantes, par photo
		self._next_id = 1

	# ---------- Association avec les étiquettes existantes ----------

	def label_faces(self) -> int:
		"""Associe aux visages détectés les noms des régions déjà présentes dans les photos.
		Retourne le nombre de régions nommées auxquelles aucun visage ne correspond."""
		by_photo: dict[str, list[FaceRecord]] = {}
		for face in self.faces:
			by_photo.setdefault(face.photo, []).append(face)
		unmatched = 0
		for photo, faces in by_photo.items():
			tags = self.photo_tags.get(photo)
			if tags is None:
				continue
			regions = [region for region in tags.regions if region["name"]]
			# Appariement glouton par recouvrement décroissant : un visage ↔ une région
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
		"""Regroupe les visages en personnes puis les nomme d'après les étiquettes existantes."""
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
			# Nom issu des régions étiquetées : l'orthographe la plus fréquente l'emporte
			spellings = Counter(face.label for face in person.faces if face.label)
			if spellings:
				person.name, person.source = spellings.most_common(1)[0][0], "tag"
		self._name_from_weak_tags(use_keywords)

	def _name_from_weak_tags(self, use_keywords: bool) -> None:
		"""Nomme les groupes restants à partir des noms sans position (PersonInImage, mots-clés).

		Un nom est attribué à un groupe s'il figure dans au moins 60 % de ses photos et s'il
		domine nettement les autres candidats. Un même nom n'est donné qu'à un seul groupe."""
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
		person = Person(identifier = self._next_id, name = name, source = source)
		self.persons[person.identifier] = person
		self._next_id += 1
		return person

	def person_by_name(self, name: str) -> Optional[Person]:
		key = name_key(name)
		for person in self.persons.values():
			if person.name and name_key(person.name) == key:
				return person
		return None

	def rename(self, person: Person, name: str) -> Person:
		"""Nomme une personne. Si ce nom existe déjà, les deux groupes sont fusionnés."""
		name = normalize_name(name)
		existing = self.person_by_name(name)
		if existing is not None and existing is not person:
			return self.merge([existing, person], keep = existing)
		person.name, person.source = name, "manual"
		return person

	def merge(self, persons: list[Person], keep: Optional[Person] = None) -> Person:
		"""Fusionne plusieurs personnes. On garde en priorité une personne nommée, puis la plus grosse."""
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
		"""Déplace un visage vers une autre personne."""
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
		"""Écarte un visage (faux positif)."""
		face.deleted = True
		source = face.person
		if source is not None:
			source.faces.remove(face)
			if not source.faces:
				self.persons.pop(source.identifier, None)
		face.person = None

	def faces_of_photo(self, photo: str) -> list[FaceRecord]:
		return [face for face in self.faces if face.photo == photo and not face.deleted]


# ======================================================================================
# Construction des métadonnées à écrire
# ======================================================================================

def build_region_info(tags: PhotoTags, items: list[tuple[str, FaceRecord]]) -> tuple[dict, dict[str, str]]:
	"""Construit la structure XMP-mwg-rs:RegionInfo en conservant les régions déjà présentes.

	Pour chaque visage nommé : si une région existante le recouvre, on la complète (nom manquant) ;
	si elle porte un autre nom, le choix actuel de l'utilisateur la remplace (personne renommée ou
	fusionnée) ; sinon on ajoute une nouvelle région « Face ».
	Retourne la structure et les noms remplacés {ancien nom: nouveau nom}."""
	existing = tags.region_info or {}
	region_list = [dict(region) for region in (existing.get("RegionList") or []) if isinstance(region, dict)]
	info = {
		"AppliedToDimensions": existing.get("AppliedToDimensions") or {"W": 1, "H": 1, "Unit": "pixel"},
		"RegionList": region_list,
	}

	def region_box(region: dict) -> Optional[tuple]:
		area = region.get("Area") or {}
		try:
			width, height = float(area["W"]), float(area["H"])
			return (float(area["X"]) - width / 2, float(area["Y"]) - height / 2, width, height)
		except (KeyError, TypeError, ValueError):
			return None

	replacements: dict[str, str] = {}
	for name, face in items:
		x, y, width, height = face.box
		match = None
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
				"X": round(x + width / 2, 6), "Y": round(y + height / 2, 6),
				"W": round(width, 6), "H": round(height, 6), "Unit": "normalized",
			},
			"Name": name,
			"Type": "Face",
		})
	return info, replacements


def check_written(tags: PhotoTags, names: list[str], regions: bool, person: bool, keywords: bool) -> bool:
	"""Vérifie, d'après les étiquettes relues, que les noms demandés figurent bien dans la photo."""
	wanted = {name_key(name) for name in names}
	present = []
	if person:
		present.append({name_key(name) for name in tags.person_in_image})
	if keywords:
		present.append({name_key(name) for name in tags.subject})
	if regions:
		present.append({name_key(region["name"]) for region in tags.regions})
	return all(wanted <= found for found in present)


# ======================================================================================
# Interface graphique
# ======================================================================================

class NameDialog(tk.Toplevel):
	"""Boîte de saisie d'un nom, avec liste déroulante des noms déjà connus."""

	def __init__(self, master: tk.Misc, title: str, prompt: str, choices: list[str], initial: str = ""):
		super().__init__(master)
		self.title(emoji(title))
		self.transient(master)
		self.resizable(False, False)
		self.result: Optional[str] = None
		self.choices = choices

		ttk.Label(self, text = prompt, padding = 8).pack()
		self.variable = tk.StringVar(value = initial)
		self.combobox = ttk.Combobox(self, textvariable = self.variable, values = choices, width = 42)
		self.combobox.pack(padx = 12)
		# La liste se restreint au fil de la frappe
		self.combobox.bind("<KeyRelease>", self.filter_choices)
		buttons = ttk.Frame(self, padding = 8)
		buttons.pack()
		ttk.Button(buttons, text = "✅ Valider", command = self.accept).pack(side = "left", padx = 4)
		ttk.Button(buttons, text = "✖️ Annuler", command = self.destroy).pack(side = "left", padx = 4)
		self.bind("<Return>", lambda event: self.accept())
		self.bind("<Escape>", lambda event: self.destroy())
		strip_widget_emoji(self)
		self.grab_set()
		self.combobox.focus_set()
		self.wait_window(self)

	def filter_choices(self, event: tk.Event) -> None:
		typed = self.variable.get().casefold()
		self.combobox.configure(values = [c for c in self.choices if typed in c.casefold()])

	def accept(self) -> None:
		name = normalize_name(self.variable.get())
		if name:
			self.result = name
			self.destroy()
		else:
			self.bell()


class PhotoViewer(tk.Toplevel):
	"""Affiche une photo entière avec le cadre et le nom de chaque visage."""

	def __init__(self, master: tk.Misc, library: Library, face: FaceRecord):
		super().__init__(master)
		self.title(emoji(f"🖼️ {face.photo}"))
		try:
			with Image.open(face.photo) as source:
				image = ImageOps.exif_transpose(source).convert("RGB")
		except Exception as error:
			messagebox.showerror(emoji("❌ Erreur"), f"Impossible d'ouvrir la photo :\n{error}", parent = master)
			self.destroy()
			return
		image.thumbnail((1100, 780))
		draw = ImageDraw.Draw(image)
		font = load_font(max(12, image.width // 55))
		for other in library.faces_of_photo(face.photo):
			x, y, width, height = other.box
			left, top = x * image.width, y * image.height
			right, bottom = (x + width) * image.width, (y + height) * image.height
			# Rouge pour le visage sélectionné, vert pour les autres
			color = "#ff3b30" if other is face else "#34c759"
			draw.rectangle([left, top, right, bottom], outline = color, width = 3)
			# Le nom est écrit sur un fond plein pour rester lisible sur les images chargées
			caption = other.person.display_name if other.person else "?"
			text_box = draw.textbbox((left, bottom + 2), caption, font = font)
			draw.rectangle(text_box, fill = color)
			draw.text((left, bottom + 2), caption, fill = "white", font = font)
		self.photo_image = ImageTk.PhotoImage(image)
		ttk.Label(self, image = self.photo_image).pack()
		self.bind("<Escape>", lambda event: self.destroy())


class UnknownNamesDialog(tk.Toplevel):
	"""Parcourt les personnes non nommées, une par une, et propose de saisir leur nom."""

	def __init__(self, app: "FaceTaggerApp"):
		super().__init__(app)
		self.app = app
		self.title(emoji("✍️ Nommer les personnes non reconnues"))
		self.transient(app)
		# Les groupes les plus gros d'abord
		unnamed = [p for p in app.library.persons.values() if not p.name and not p.ignored]
		self.pending = sorted(unnamed, key = lambda p: -len(p.faces))
		self.index = 0
		self.person: Optional[Person] = None
		self.images: list[ImageTk.PhotoImage] = []

		self.header = ttk.Label(self, padding = 8, font = ("TkDefaultFont", 11, "bold"))
		self.header.pack()
		self.gallery = ttk.Frame(self, padding = 8)
		self.gallery.pack()
		ttk.Label(self, text = "Nom de cette personne (ou nom déjà utilisé pour fusionner) :").pack()
		self.variable = tk.StringVar()
		self.combobox = ttk.Combobox(self, textvariable = self.variable, width = 44)
		self.combobox.pack(padx = 12, pady = 4)
		self.combobox.bind("<KeyRelease>", self.filter_choices)
		buttons = ttk.Frame(self, padding = 8)
		buttons.pack()
		ttk.Button(buttons, text = "✅ Valider", command = self.validate).pack(side = "left", padx = 4)
		ttk.Button(buttons, text = "⏭️ Passer", command = self.skip).pack(side = "left", padx = 4)
		ttk.Button(buttons, text = "🙈 Ignorer cette personne", command = self.ignore).pack(side = "left", padx = 4)
		ttk.Button(buttons, text = "⏹️ Terminer", command = self.destroy).pack(side = "left", padx = 4)
		self.bind("<Return>", lambda event: self.validate())
		self.bind("<Escape>", lambda event: self.destroy())
		strip_widget_emoji(self)
		self.grab_set()
		self.show_current()

	def known_names(self) -> list[str]:
		names = {p.name for p in self.app.library.persons.values() if p.name}
		return sorted(names, key = str.casefold)

	def filter_choices(self, event: tk.Event) -> None:
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
			f"{len(self.person.faces)} visage(s) dans {len(self.person.photos)} photo(s)"
		))
		for child in self.gallery.winfo_children():
			child.destroy()
		self.images.clear()
		best = sorted(self.person.faces, key = lambda f: -f.score)[:12]
		thumbnails = self.app.database.thumbnails([face.face_id for face in best])
		for position, face in enumerate(best):
			image = ImageTk.PhotoImage(decode_thumbnail(thumbnails.get(face.face_id)))
			self.images.append(image)
			ttk.Label(self.gallery, image = image).grid(row = position // 6, column = position % 6, padx = 3, pady = 3)
		self.variable.set("")
		self.combobox.configure(values = self.known_names())
		self.combobox.focus_set()

	def validate(self) -> None:
		name = normalize_name(self.variable.get())
		if not name or self.person is None:
			self.bell()
			return
		merged = self.app.library.rename(self.person, name)
		if merged is not self.person:
			self.app.log(f"🔗 Groupe fusionné avec « {merged.name} »")
		else:
			self.app.log(f"✍️ Personne nommée « {name} »")
		self.app.refresh_persons()
		self.index += 1
		self.show_current()

	def skip(self) -> None:
		self.index += 1
		self.show_current()

	def ignore(self) -> None:
		if self.person is not None:
			self.person.ignored = True
			self.app.refresh_persons()
		self.index += 1
		self.show_current()


class FaceTaggerApp(tk.Tk):
	"""Fenêtre principale de l'application."""

	def __init__(self, options: argparse.Namespace):
		# Avant super().__init__() : l'identité dans la barre des tâches doit précéder la création de la fenêtre
		set_windows_app_id()
		super().__init__()
		self.options = options
		self.title(emoji("👥 Face Tagger — visages par personne"))
		# Icône de la fenêtre, et par défaut de toutes les boîtes de dialogue (Toplevel). Linux et macOS refusent
		# les fichiers .ico (TclError) : sans icône, l'application démarre quand même, de même si le fichier est
		# introuvable.
		try:
			icon_file = resource_path("face_tagger.ico")
			self.iconbitmap(default = icon_file)
		except tk.TclError:
			pass
		self.geometry("1300x820")
		self.data_directory = Path(options.data_directory).expanduser()
		self.exiftool = ExifTool(options.exiftool)
		self.library = Library()
		self.database = Database(Path(options.database).expanduser())
		self.events: queue.Queue = queue.Queue()          # Messages des threads de travail vers l'interface
		self.cancel_event = threading.Event()
		self.worker: Optional[threading.Thread] = None
		self.current_person: Optional[Person] = None
		# Références aux images Tk (sinon le ramasse-miettes les supprime et elles disparaissent)
		self.icon_cache: dict[int, ImageTk.PhotoImage] = {}      # Icônes de la liste des personnes, par visage
		self.face_images: list[ImageTk.PhotoImage] = []          # Vignettes de la page affichée
		self.face_labels: list[ttk.Label] = []
		self.sorted_faces: list[FaceRecord] = []                 # Visages de la personne affichée, du plus net au moins net
		self.page = 0
		self.grid_columns = 0

		# Variables liées aux widgets
		self.directory_var = tk.StringVar(value = options.directory or "")
		self.threshold_var = tk.DoubleVar(value = options.threshold)
		self.keywords_var = tk.BooleanVar(value = False)
		self.write_regions_var = tk.BooleanVar(value = True)
		self.write_person_var = tk.BooleanVar(value = True)
		self.write_keywords_var = tk.BooleanVar(value = True)
		self.backup_var = tk.BooleanVar(value = not options.no_backup)
		self.page_size_var = tk.StringVar(value = str(options.page_size))
		self.status_var = tk.StringVar(value = emoji("👋 Choisissez un dossier de photos puis cliquez sur « Analyser »."))

		self.build_widgets()
		strip_widget_emoji(self)
		self.protocol("WM_DELETE_WINDOW", self.on_close)
		self.after(100, self.poll_events)
		if not self.exiftool.available:
			self.log("⚠️ ExifTool est introuvable : la lecture et l'écriture des métadonnées sont désactivées.")
			self.log("👉 Fedora : sudo dnf install perl-Image-ExifTool — Debian/Ubuntu : sudo apt install libimage-exiftool-perl")
		stats = self.database.statistics()
		self.log(f"🗄️ Base de données : {self.database.path} ({stats['photos']} photo(s), {stats['faces']} visage(s), "
				f"{stats['backups']} sauvegarde(s) *_original suivie(s)).")

	# ---------- Construction de la fenêtre ----------

	def build_widgets(self) -> None:
		ttk.Style(self).configure("Treeview", rowheight = TREE_ROW_HEIGHT)

		# --- Barre du haut : dossier et lancement de l'analyse ---
		top = ttk.Frame(self, padding = 6)
		top.pack(fill = "x")
		ttk.Label(top, text = "📁 Dossier :").pack(side = "left")
		ttk.Entry(top, textvariable = self.directory_var).pack(side = "left", fill = "x", expand = True, padx = 6)
		ttk.Button(top, text = "Parcourir…", command = self.choose_directory).pack(side = "left")
		self.scan_button = ttk.Button(top, text = "🔍 Analyser", command = self.start_scan)
		self.scan_button.pack(side = "left", padx = 6)
		self.stop_button = ttk.Button(top, text = "⏹️ Arrêter", command = self.cancel_event.set, state = "disabled")
		self.stop_button.pack(side = "left")

		# --- Réglages du regroupement ---
		settings = ttk.Frame(self, padding = (6, 0))
		settings.pack(fill = "x")
		ttk.Label(settings, text = "🎚️ Seuil de similarité :").pack(side = "left")
		self.threshold_label = ttk.Label(settings, text = f"{self.threshold_var.get():.2f}", width = 5)
		ttk.Scale(
			settings, from_ = 0.20, to = 0.70, variable = self.threshold_var, length = 220,
			command = lambda value: self.threshold_label.configure(text = f"{float(value):.2f}"),
		).pack(side = "left", padx = 6)
		self.threshold_label.pack(side = "left")
		ttk.Label(settings, text = "(plus bas = groupes plus larges)").pack(side = "left", padx = 4)
		ttk.Checkbutton(
			settings, text = "Utiliser aussi les mots-clés comme noms candidats", variable = self.keywords_var,
		).pack(side = "left", padx = 16)
		self.regroup_button = ttk.Button(settings, text = "🔄 Regrouper à nouveau", command = self.regroup)
		self.regroup_button.pack(side = "left")

		# --- Zone centrale : personnes à gauche, visages de la personne choisie à droite ---
		panes = ttk.PanedWindow(self, orient = "horizontal")
		panes.pack(fill = "both", expand = True, padx = 6, pady = 6)

		left = ttk.Frame(panes)
		panes.add(left, weight = 1)
		self.tree = ttk.Treeview(left, columns = ("faces", "photos", "source"), selectmode = "extended")
		self.tree.heading("#0", text = "Personne")
		self.tree.heading("faces", text = "Visages")
		self.tree.heading("photos", text = "Photos")
		self.tree.heading("source", text = "Origine du nom")
		self.tree.column("#0", width = 220)
		self.tree.column("faces", width = 60, anchor = "center")
		self.tree.column("photos", width = 60, anchor = "center")
		self.tree.column("source", width = 110)
		tree_scroll = ttk.Scrollbar(left, orient = "vertical", command = self.tree.yview)
		self.tree.configure(yscrollcommand = tree_scroll.set)
		person_buttons = ttk.Frame(left)
		person_buttons.pack(side = "bottom", fill = "x", pady = 4)
		ttk.Button(person_buttons, text = "✏️ Renommer…", command = self.rename_selected).pack(side = "left", padx = 2)
		ttk.Button(person_buttons, text = "🔗 Fusionner", command = self.merge_selected).pack(side = "left", padx = 2)
		ttk.Button(person_buttons, text = "🙈 Ignorer / réactiver", command = self.toggle_ignore).pack(side = "left", padx = 2)
		ttk.Button(person_buttons, text = "🧩 Nommer les inconnus…", command = self.name_unknowns).pack(side = "left", padx = 2)
		tree_scroll.pack(side = "right", fill = "y")
		self.tree.pack(side = "left", fill = "both", expand = True)
		self.tree.bind("<<TreeviewSelect>>", self.on_select)
		self.tree.bind("<Double-Button-1>", lambda event: self.rename_selected())

		right = ttk.Frame(panes)
		panes.add(right, weight = 3)
		self.person_title = ttk.Label(right, text = "", font = ("TkDefaultFont", 12, "bold"))
		self.person_title.pack(anchor = "w", padx = 4)

		# Navigation entre les pages : seules les vignettes de la page affichée sont chargées en mémoire
		navigation = ttk.Frame(right)
		navigation.pack(fill = "x", padx = 4, pady = 2)
		self.first_button = ttk.Button(navigation, text = "«", width = 3, command = lambda: self.go_to_page(0))
		self.previous_button = ttk.Button(navigation, text = "‹", width = 3, command = lambda: self.go_to_page(self.page - 1))
		self.page_label = ttk.Label(navigation, text = "Page 1 / 1", width = 14, anchor = "center")
		self.next_button = ttk.Button(navigation, text = "›", width = 3, command = lambda: self.go_to_page(self.page + 1))
		self.last_button = ttk.Button(navigation, text = "»", width = 3, command = lambda: self.go_to_page(10 ** 9))
		for widget in (self.first_button, self.previous_button, self.page_label, self.next_button, self.last_button):
			widget.pack(side = "left", padx = 1)
		self.range_label = ttk.Label(navigation, text = "")
		self.range_label.pack(side = "left", padx = 10)
		self.page_size_box = ttk.Combobox(
			navigation, textvariable = self.page_size_var, values = ("50", "100", "200", "500"), width = 5, state = "readonly",
		)
		self.page_size_box.pack(side = "right")
		ttk.Label(navigation, text = "Par page :").pack(side = "right", padx = 4)
		self.page_size_box.bind("<<ComboboxSelected>>", lambda event: self.go_to_page(0))

		ttk.Label(right, text = "Double-clic : ouvrir la photo — clic droit : déplacer ou écarter le visage").pack(anchor = "w", padx = 4)
		self.canvas = tk.Canvas(right, highlightthickness = 0)
		canvas_scroll = ttk.Scrollbar(right, orient = "vertical", command = self.canvas.yview)
		self.canvas.configure(yscrollcommand = canvas_scroll.set)
		canvas_scroll.pack(side = "right", fill = "y")
		self.canvas.pack(side = "left", fill = "both", expand = True)
		self.faces_frame = ttk.Frame(self.canvas)
		self.canvas.create_window((0, 0), window = self.faces_frame, anchor = "nw")
		self.faces_frame.bind("<Configure>", lambda event: self.canvas.configure(scrollregion = self.canvas.bbox("all")))
		self.canvas.bind("<Configure>", lambda event: self.layout_faces())
		# Molette de la souris (Windows / macOS : <MouseWheel>, Linux : boutons 4 et 5)
		self.canvas.bind("<MouseWheel>", lambda event: self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"))
		self.canvas.bind("<Button-4>", lambda event: self.canvas.yview_scroll(-1, "units"))
		self.canvas.bind("<Button-5>", lambda event: self.canvas.yview_scroll(1, "units"))

		# --- Bas de fenêtre : écriture des métadonnées, progression et journal ---
		bottom = ttk.Frame(self, padding = 6)
		bottom.pack(fill = "x")
		ttk.Label(bottom, text = "💾 Écrire :").pack(side = "left")
		ttk.Checkbutton(bottom, text = "régions de visages (MWG)", variable = self.write_regions_var).pack(side = "left", padx = 4)
		ttk.Checkbutton(bottom, text = "PersonInImage", variable = self.write_person_var).pack(side = "left", padx = 4)
		ttk.Checkbutton(bottom, text = "mots-clés (XMP, EXIF)", variable = self.write_keywords_var).pack(side = "left", padx = 4)
		ttk.Checkbutton(bottom, text = "sauvegarde *_original", variable = self.backup_var).pack(side = "left", padx = 12)
		self.write_button = ttk.Button(bottom, text = "💾 Écrire dans les métadonnées", command = self.start_write)
		self.write_button.pack(side = "right")
		self.restore_button = ttk.Button(bottom, text = "♻️ Restaurer les sauvegardes…", command = self.start_restore)
		self.restore_button.pack(side = "right", padx = 6)

		self.progress = ttk.Progressbar(self, mode = "determinate")
		self.progress.pack(fill = "x", padx = 6)
		ttk.Label(self, textvariable = self.status_var, padding = (6, 2)).pack(fill = "x")
		self.log_text = ScrolledText(self, height = 8, state = "disabled", wrap = "word")
		self.log_text.pack(fill = "x", padx = 6, pady = (0, 6))

	# ---------- Journal et communication avec les threads ----------

	def log(self, message: str) -> None:
		"""Ajoute un message au journal de la fenêtre et à la console."""
		message = emoji(message)
		print(message, flush = True)
		self.log_text.configure(state = "normal")
		self.log_text.insert("end", message + "\n")
		self.log_text.see("end")
		self.log_text.configure(state = "disabled")

	def post(self, *event) -> None:
		"""Envoie un événement vers le thread de l'interface (appelable depuis un thread de travail)."""
		self.events.put(event)

	def poll_events(self) -> None:
		"""Traite les événements envoyés par les threads de travail (appelée toutes les 100 ms)."""
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
					self.library.photo_tags.update(event[1])
				elif kind == "restored":
					self.after_restore(event[1])
				elif kind == "done":
					self.set_busy(False)
		except queue.Empty:
			pass
		self.after(100, self.poll_events)

	def set_busy(self, busy: bool) -> None:
		"""Active ou désactive les boutons pendant une analyse ou une écriture."""
		state = "disabled" if busy else "normal"
		for button in (self.scan_button, self.write_button, self.regroup_button, self.restore_button):
			button.configure(state = state)
		self.stop_button.configure(state = "normal" if busy else "disabled")

	def on_close(self) -> None:
		self.cancel_event.set()
		self.database.close()
		self.destroy()

	# ---------- Analyse des photos (thread de travail) ----------

	def choose_directory(self) -> None:
		chosen = filedialog.askdirectory(initialdir = self.directory_var.get() or str(Path.home()))
		if chosen:
			self.directory_var.set(chosen)

	def start_scan(self) -> None:
		directory = os.path.abspath(os.path.expanduser(self.directory_var.get().strip()))
		if not os.path.isdir(directory):
			messagebox.showerror(emoji("❌ Dossier introuvable"), "Choisissez un dossier de photos valide.")
			return
		self.cancel_event.clear()
		self.set_busy(True)
		self.progress.configure(value = 0)
		# Les valeurs des widgets sont lues ici, dans le thread de l'interface
		threshold, use_keywords = float(self.threshold_var.get()), self.keywords_var.get()
		self.worker = threading.Thread(target = self.scan_worker, args = (directory, threshold, use_keywords), daemon = True)
		self.worker.start()

	def create_engine(self) -> FaceEngine:
		"""Charge (et au besoin télécharge) les modèles. N'est appelée que si une photo doit être analysée."""
		detection_model, recognition_model = ensure_models(
			self.data_directory / "models", lambda message: self.post("log", message),
		)
		return FaceEngine(
			detection_model, recognition_model, self.options.max_image_size,
			self.options.min_face_size, self.options.detection_score,
		)

	def scan_worker(self, directory: str, threshold: float, use_keywords: bool) -> None:
		"""Détecte les visages, lit les étiquettes existantes puis regroupe (exécuté hors du thread Tk).
		Ce qui est déjà en base pour une photo inchangée n'est ni ré-analysé ni relu."""
		try:
			self.post("log", f"📂 Exploration de {directory}…")
			photos = list_photos(directory)
			if not photos:
				self.post("log", "⚠️ Aucune photo trouvée dans ce dossier.")
				return
			self.post("log", f"🖼️ {len(photos)} photos trouvées.")

			# Si les paramètres de détection ont changé, les visages mémorisés ne sont plus valables
			signature = f"{self.options.max_image_size}|{self.options.min_face_size}|{self.options.detection_score}"
			if self.database.reset_analysis_if_needed(signature):
				self.post("log", "🗄️ Paramètres de détection modifiés : les photos seront ré-analysées.")
			known = self.database.signatures()

			engine: Optional[FaceEngine] = None
			processed: set[str] = set()
			analysed = reused = failures = 0
			for index, photo in enumerate(photos, start = 1):
				if self.cancel_event.is_set():
					self.post("log", "⏹️ Analyse interrompue : les photos déjà traitées sont conservées.")
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
						self.post("log", f"⚠️ Photo ignorée ({os.path.basename(photo)}) : {error}")
						continue
					self.database.store_analysis(photo, stat.st_mtime_ns, stat.st_size, found)
					analysed += 1
				processed.add(photo)
			self.post("log", f"🗄️ {reused} photo(s) reprise(s) depuis la base, {analysed} analysée(s), {failures} en échec.")

			# Les visages viennent de la base (sans les vignettes, lues plus tard à la demande)
			library = Library()
			for face_id, photo, x, y, width, height, score, blob in self.database.load_faces(processed):
				library.faces.append(FaceRecord(
					photo = photo, box = (x, y, width, height), score = score,
					embedding = np.frombuffer(blob, dtype = np.float32), face_id = face_id,
				))
			self.post("log", f"✅ {len(library.faces)} visages au total.")

			# Étiquettes déjà présentes : d'abord depuis la base, ExifTool ne lit que les photos manquantes
			if self.exiftool.available and library.faces:
				with_faces = sorted({face.photo for face in library.faces})
				stored = self.database.tags(with_faces)
				missing = [photo for photo in with_faces if photo not in stored]
				self.post("log", f"🗄️ Étiquettes : {len(stored)} reprises depuis la base, {len(missing)} à lire avec ExifTool.")
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
				self.post("log", f"🏷️ {labeled} visage(s) rattaché(s) à un nom déjà présent dans les photos.")
				if unmatched:
					self.post("log", f"ℹ️ {unmatched} région(s) nommée(s) sans visage détecté correspondant.")

			library.rebuild(threshold, use_keywords)
			named = sum(1 for person in library.persons.values() if person.name)
			self.post("log", f"👥 {len(library.persons)} personne(s) regroupée(s), dont {named} nommée(s) automatiquement.")
			self.post("library", library)
			self.post("progress", 1, 1, f"✅ Analyse terminée : {len(library.persons)} personne(s).")
		except Exception as error:
			self.post("log", f"❌ Erreur : {error}")
		finally:
			self.post("done")

	def regroup(self) -> None:
		"""Refait le regroupement avec le seuil courant, sans ré-analyser les photos."""
		if not self.library.faces:
			return
		if not messagebox.askyesno(
			emoji("🔄 Regrouper"), "Le regroupement sera refait : les noms saisis et les fusions manuelles seront perdus. Continuer ?",
		):
			return
		self.library.rebuild(float(self.threshold_var.get()), self.keywords_var.get())
		self.refresh_persons()
		self.log(f"🔄 Regroupement refait avec le seuil {self.threshold_var.get():.2f} : {len(self.library.persons)} personne(s).")

	# ---------- Liste des personnes ----------

	def refresh_persons(self, select: Optional[list[int]] = None) -> None:
		"""Reconstruit la liste des personnes (nommées d'abord, puis inconnues par taille décroissante)."""
		if select is None:
			select = [int(i) for i in self.tree.selection()]
		self.tree.delete(*self.tree.get_children())
		if len(self.icon_cache) > ICON_CACHE_SIZE:
			self.icon_cache.clear()                  # Évite une croissance sans limite de la mémoire
		persons = sorted(self.library.persons.values(), key = lambda p: (p.name is None, (p.name or "").casefold(), -len(p.faces)))
		# Une icône par personne : son visage le plus net. Les vignettes manquantes sont lues par lots.
		bests = {person.identifier: max(person.faces, key = lambda f: f.score) for person in persons}
		thumbnails = self.database.thumbnails([face.face_id for face in bests.values() if face.face_id not in self.icon_cache])
		for person in persons:
			best = bests[person.identifier]
			icon = self.icon_cache.get(best.face_id)
			if icon is None:
				size = TREE_ROW_HEIGHT - 4
				icon = ImageTk.PhotoImage(decode_thumbnail(thumbnails.get(best.face_id)).resize((size, size)))
				self.icon_cache[best.face_id] = icon
			label = emoji(("🙈 " if person.ignored else "") + (person.name or f"❓ Inconnu n°{person.identifier}"))
			origin = {"tag": "🏷️ étiquette", "manual": "✍️ saisi"}.get(person.source, "")
			self.tree.insert(
				"", "end", iid = str(person.identifier), text = " " + label, image = icon,
				values = (len(person.faces), len(person.photos), emoji(origin)),
			)
		still_there = [str(i) for i in select if i in self.library.persons]
		if not still_there and persons:
			still_there = [str(persons[0].identifier)]   # Par défaut, on affiche la première personne
		if still_there:
			self.tree.selection_set(still_there)
		else:
			self.show_person(None)

	def selected_persons(self) -> list[Person]:
		return [self.library.persons[int(i)] for i in self.tree.selection() if int(i) in self.library.persons]

	def on_select(self, event: Optional[tk.Event] = None) -> None:
		selected = self.selected_persons()
		# La page courante est conservée tant que la personne affichée ne change pas
		self.show_person(selected[0] if len(selected) == 1 else None, keep_page = True)
		if len(selected) > 1:
			self.person_title.configure(text = emoji(f"👥 {len(selected)} personnes sélectionnées"))

	def page_size(self) -> int:
		"""Nombre de vignettes par page (réglable dans l'interface)."""
		try:
			return max(10, int(self.page_size_var.get()))
		except (ValueError, tk.TclError):
			return 100

	def show_person(self, person: Optional[Person], keep_page: bool = False) -> None:
		"""Affiche une personne : ses visages sont triés du plus net au moins net, puis découpés en pages."""
		changed = person is not self.current_person
		self.current_person = person
		self.sorted_faces = sorted(person.faces, key = lambda f: -f.score) if person is not None else []
		if changed or not keep_page:
			self.page = 0
		self.render_page()

	def go_to_page(self, page: int) -> None:
		"""Affiche la page demandée (ramenée dans les limites existantes)."""
		self.page = page
		self.render_page()

	def render_page(self) -> None:
		"""Construit les vignettes de la page courante. Les images des autres pages sont libérées :
		la mémoire utilisée dépend de la taille de la page, pas du nombre de visages de la personne."""
		for child in self.faces_frame.winfo_children():
			child.destroy()
		self.face_images.clear()
		self.face_labels.clear()
		person = self.current_person
		if person is None:
			self.person_title.configure(text = "")
			self.page_label.configure(text = "Page – / –")
			self.range_label.configure(text = "")
			for button in (self.first_button, self.previous_button, self.next_button, self.last_button):
				button.configure(state = "disabled")
			return

		size = self.page_size()
		total = len(self.sorted_faces)
		pages = max(1, math.ceil(total / size))
		self.page = min(max(self.page, 0), pages - 1)
		chunk = self.sorted_faces[self.page * size:(self.page + 1) * size]

		self.person_title.configure(text = emoji(
			f"{'🙈 ' if person.ignored else '👤 '}{person.display_name} — {total} visage(s), {len(person.photos)} photo(s)"
		))
		self.page_label.configure(text = f"Page {self.page + 1} / {pages}")
		self.range_label.configure(text = f"{self.page * size + 1 if chunk else 0}–{self.page * size + len(chunk)} sur {total}")
		for button in (self.first_button, self.previous_button):
			button.configure(state = "normal" if self.page > 0 else "disabled")
		for button in (self.next_button, self.last_button):
			button.configure(state = "normal" if self.page < pages - 1 else "disabled")

		# Les vignettes de la page sont lues dans la base en une seule requête
		thumbnails = self.database.thumbnails([face.face_id for face in chunk])
		for face in chunk:
			image = ImageTk.PhotoImage(decode_thumbnail(thumbnails.get(face.face_id)))
			self.face_images.append(image)
			label = ttk.Label(self.faces_frame, image = image, text = os.path.basename(face.photo)[:16], compound = "top")
			label.bind("<Double-Button-1>", lambda event, f = face: self.open_photo(f))
			label.bind("<Button-3>", lambda event, f = face: self.face_menu(event, f))
			label.bind("<Button-2>", lambda event, f = face: self.face_menu(event, f))   # Clic droit sous macOS
			label.bind("<MouseWheel>", lambda event: self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"))
			label.bind("<Button-4>", lambda event: self.canvas.yview_scroll(-1, "units"))
			label.bind("<Button-5>", lambda event: self.canvas.yview_scroll(1, "units"))
			self.face_labels.append(label)
		self.layout_faces(force = True)
		self.canvas.yview_moveto(0)

	def layout_faces(self, force: bool = False) -> None:
		"""Place les vignettes en grille. N'est refaite que si le nombre de colonnes change."""
		columns = max(1, self.canvas.winfo_width() // (THUMBNAIL_SIZE + 24))
		if columns == self.grid_columns and not force:
			return
		self.grid_columns = columns
		for position, label in enumerate(self.face_labels):
			label.grid(row = position // columns, column = position % columns, padx = 4, pady = 4)

	# ---------- Actions sur les personnes et les visages ----------

	def known_names(self) -> list[str]:
		return sorted({p.name for p in self.library.persons.values() if p.name}, key = str.casefold)

	def rename_selected(self) -> None:
		selected = self.selected_persons()
		if len(selected) != 1:
			return
		person = selected[0]
		dialog = NameDialog(self, "✏️ Renommer", "Nom de la personne (un nom existant fusionne les deux groupes) :",
							self.known_names(), initial = person.name or "")
		if dialog.result:
			merged = self.library.rename(person, dialog.result)
			self.log(f"✍️ « {merged.name} » : {len(merged.faces)} visage(s)")
			self.refresh_persons(select = [merged.identifier])

	def merge_selected(self) -> None:
		persons = self.selected_persons()
		if len(persons) < 2:
			messagebox.showinfo(emoji("🔗 Fusionner"), "Sélectionnez au moins deux personnes (Ctrl ou Maj + clic).")
			return
		if len({name_key(p.name) for p in persons if p.name}) > 1 and not messagebox.askyesno(
			emoji("🔗 Fusionner"), "Ces personnes portent des noms différents. Fusionner quand même ?",
		):
			return
		merged = self.library.merge(persons)
		self.log(f"🔗 {len(persons)} groupes fusionnés en « {merged.display_name} »")
		self.refresh_persons(select = [merged.identifier])

	def toggle_ignore(self) -> None:
		for person in self.selected_persons():
			person.ignored = not person.ignored
		self.refresh_persons()
		self.show_person(self.current_person, keep_page = True)

	def name_unknowns(self) -> None:
		if not any(not p.name and not p.ignored for p in self.library.persons.values()):
			messagebox.showinfo(emoji("🎉 Rien à faire"), "Toutes les personnes sont déjà nommées ou ignorées.")
			return
		UnknownNamesDialog(self)
		self.refresh_persons()

	def open_photo(self, face: FaceRecord) -> None:
		PhotoViewer(self, self.library, face)

	def face_menu(self, event: tk.Event, face: FaceRecord) -> None:
		"""Menu contextuel d'un visage."""
		menu = tk.Menu(self, tearoff = False)
		menu.add_command(label = emoji("🖼️ Ouvrir la photo"), command = lambda: self.open_photo(face))
		menu.add_command(label = emoji("↪️ Déplacer vers une personne…"), command = lambda: self.move_face(face))
		menu.add_command(label = emoji("✂️ Détacher dans un nouveau groupe"), command = lambda: self.detach_face(face))
		menu.add_command(label = emoji("🗑️ Ce n'est pas un visage (écarter)"), command = lambda: self.delete_face(face))
		menu.tk_popup(event.x_root, event.y_root)

	def move_face(self, face: FaceRecord) -> None:
		dialog = NameDialog(self, "↪️ Déplacer le visage", "Nom de la personne de destination :", self.known_names())
		if not dialog.result:
			return
		target = self.library.person_by_name(dialog.result) or self.library.new_person(dialog.result, "manual")
		origin = face.person
		self.library.move_face(face, target)
		self.refresh_persons(select = [origin.identifier if origin and origin.identifier in self.library.persons else target.identifier])

	def detach_face(self, face: FaceRecord) -> None:
		person = self.library.new_person()
		self.library.move_face(face, person)
		self.refresh_persons(select = [person.identifier])

	def delete_face(self, face: FaceRecord) -> None:
		origin = face.person
		self.library.delete_face(face)
		self.refresh_persons(select = [origin.identifier] if origin else [])

	# ---------- Écriture des métadonnées (thread de travail) ----------

	def collect_updates(self) -> dict[str, list[tuple[str, FaceRecord]]]:
		"""Regroupe par photo les (nom, visage) à écrire : personnes nommées et non ignorées."""
		updates: dict[str, list[tuple[str, FaceRecord]]] = {}
		for person in self.library.persons.values():
			if person.name and not person.ignored:
				for face in person.faces:
					updates.setdefault(face.photo, []).append((person.name, face))
		return updates

	def start_write(self) -> None:
		if not self.exiftool.available:
			messagebox.showerror(emoji("❌ ExifTool manquant"), "Installez ExifTool pour écrire les métadonnées.")
			return
		if not (self.write_regions_var.get() or self.write_person_var.get() or self.write_keywords_var.get()):
			messagebox.showinfo(emoji("💾 Écriture"), "Cochez au moins un type de métadonnées à écrire.")
			return
		updates = self.collect_updates()
		if not updates:
			messagebox.showinfo(emoji("💾 Écriture"), "Aucune personne nommée à écrire. Nommez d'abord des personnes.")
			return
		names = {name for items in updates.values() for name, _ in items}
		backup_note = "\nUne copie *_original de chaque photo sera conservée." if self.backup_var.get() else "\n⚠️ Les photos seront modifiées sans copie de sauvegarde."
		if not messagebox.askyesno(
			emoji("💾 Écriture"), emoji(f"Écrire {len(names)} nom(s) dans les métadonnées de {len(updates)} photo(s) ?{backup_note}"),
		):
			return
		self.set_busy(True)
		write_options = (self.write_regions_var.get(), self.write_person_var.get(), self.write_keywords_var.get(), self.backup_var.get())
		self.worker = threading.Thread(target = self.write_worker, args = (updates, write_options), daemon = True)
		self.worker.start()

	def write_worker(self, updates: dict[str, list[tuple[str, FaceRecord]]], write_options: tuple) -> None:
		"""Écrit les métadonnées par lots (exécuté hors du thread Tk). Après chaque lot, les étiquettes sont
		relues pour vérifier l'écriture, puis la base est mise à jour (signature, étiquettes, historique)."""
		write_regions, write_person, write_keywords, backup = write_options
		fields = [label for flag, label in ((write_regions, "regions"), (write_person, "PersonInImage"), (write_keywords, "keywords")) if flag]
		try:
			photos = list(updates)
			self.post("log", f"💾 Écriture des métadonnées de {len(photos)} photo(s)…")
			problems = verified_count = backups_count = 0
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
							self.post("log", f"✏️ {os.path.basename(photo)} : « {old} » remplacé par « {new} »")
					# Les anciens noms remplacés sont retirés des listes (sauf s'ils sont aussi voulus ailleurs)
					removed = {name_key(old) for old in replacements} - {name_key(name) for name in names}

					def kept(existing: list[str]) -> list[str]:
						return [name for name in existing if name_key(name) not in removed]

					if write_person:
						record["XMP-iptcExt:PersonInImage"] = merge_unique(kept(tags.person_in_image), names)
					if write_keywords:
						# XMP est en UTF-8 par norme. L'IPTC n'est volontairement pas modifié : son jeu de caractères
						# n'est pas toujours déclaré et une écriture risquerait de corrompre les accents existants.
						record["XMP-dc:Subject"] = merge_unique(kept(tags.subject), names)
						record["XPKeywords"] = ";".join(merge_unique(kept(tags.xp_keywords), names))
					records.append(record)

				# 2. Écriture par ExifTool (la 1re version de chaque photo est conservée dans *_original)
				for message in self.exiftool.write_tags(records, backup):
					if "updated" in message or "unchanged" in message:
						self.post("log", f"✅ {message}")
					else:
						problems += 1
						self.post("log", f"⚠️ {message}")

				# 3. Relecture : vérification des noms, puis mise à jour de la base
				entries = self.exiftool.read_tags(batch)
				fresh: dict[str, dict] = {}
				new_tags: dict[str, PhotoTags] = {}
				for photo in batch:
					entry = entries.get(path_key(photo))
					if entry is None:
						problems += 1
						self.post("log", f"⚠️ Relecture impossible : {os.path.basename(photo)}")
						continue
					names = merge_unique([name for name, _ in updates[photo]])
					tags = parse_photo_tags(entry)
					if check_written(tags, names, write_regions, write_person, write_keywords):
						verified = True
						verified_count += 1
					else:
						verified = False
						problems += 1
						self.post("log", f"⚠️ Noms attendus absents après écriture : {os.path.basename(photo)} (conflit avec une étiquette existante ?)")
					try:
						# La taille du fichier a changé mais pas ses pixels : on met la signature à jour
						# et les visages mémorisés restent valables.
						stat = os.stat(photo)
						self.database.update_signature(photo, stat.st_mtime_ns, stat.st_size)
					except OSError:
						pass
					backup_path = photo + restore_originals.BACKUP_SUFFIX
					has_backup = os.path.exists(backup_path)
					backups_count += int(has_backup)
					self.database.record_modification(photo, names, fields, verified, backup_path if has_backup else None)
					fresh[photo] = entry
					new_tags[photo] = tags
				self.database.store_tags(fresh)
				self.post("tags", new_tags)

			self.post("progress", len(photos), len(photos), f"✅ Écriture terminée ({problems} avertissement(s)).")
			self.post("log", f"🎉 Terminé : {verified_count}/{len(photos)} photo(s) vérifiée(s) après relecture, {problems} avertissement(s).")
			if backups_count:
				self.post("log", f"📦 {backups_count} sauvegarde(s) *_original disponible(s) — bouton « Restaurer les sauvegardes » ou restore_originals.py.")
		except Exception as error:
			self.post("log", f"❌ Erreur pendant l'écriture : {error}")
		finally:
			self.post("done")

	# ---------- Restauration des sauvegardes *_original ----------

	def start_restore(self) -> None:
		"""Restaure d'un coup toutes les photos d'un dossier à partir de leurs sauvegardes *_original."""
		directory = filedialog.askdirectory(
			initialdir = self.directory_var.get() or str(Path.home()), title = "Dossier contenant les fichiers *_original",
		)
		if not directory:
			return
		backups = restore_originals.find_backups(os.path.abspath(directory))
		if not backups:
			messagebox.showinfo(emoji("ℹ️ Restauration"), "Aucune sauvegarde *_original trouvée dans ce dossier.")
			return
		if not messagebox.askyesno(
			emoji("♻️ Restauration"),
			emoji(f"Remplacer {len(backups)} photo(s) par leur version d'origine ?\n"
				"Les noms écrits dans leurs métadonnées seront perdus, et la liste des personnes sera vidée "
				"si elle contient ces photos (relancez ensuite « Analyser »)."),
		):
			return
		self.set_busy(True)
		self.worker = threading.Thread(target = self.restore_worker, args = (backups,), daemon = True)
		self.worker.start()

	def restore_worker(self, backups: list[tuple[str, str]]) -> None:
		"""Restaure les photos puis met la base à jour (exécuté hors du thread Tk)."""
		try:
			self.post("log", f"♻️ Restauration de {len(backups)} photo(s)…")
			done, failed = restore_originals.restore_backups(
				backups, database = self.database, log = lambda message: self.post("log", message),
			)
			self.post("log", f"🎉 Terminé : {done} photo(s) restaurée(s), {failed} échec(s).")
			self.post("restored", {target for _, target in backups})
		except Exception as error:
			self.post("log", f"❌ Erreur pendant la restauration : {error}")
		finally:
			self.post("done")

	def after_restore(self, targets: set[str]) -> None:
		"""Les étiquettes en mémoire ne correspondent plus aux fichiers restaurés : on repart d'une liste vide."""
		if targets & {face.photo for face in self.library.faces}:
			self.library = Library()
			self.refresh_persons()
			self.log("🔄 Liste des personnes vidée : relancez « Analyser » (rapide : seules les étiquettes des photos restaurées sont relues).")


# ======================================================================================
# Ligne de commande
# ======================================================================================

def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description = "👥 Regroupe les visages d'un dossier de photos par personne et écrit les noms dans les métadonnées.")
	parser.add_argument("--directory", default = "", help = "📁 Dossier de photos à ouvrir au démarrage")
	parser.add_argument("--data-directory", default = str(Path.home() / ".face_tagger"), help = "🗄️ Dossier des modèles ONNX et du cache")
	parser.add_argument("--database", default = "", help = "🗄️ Fichier de la base SQLite (défaut : <data-directory>/face_tagger.sqlite)")
	parser.add_argument("--page-size", type = int, default = 100, help = "📄 Nombre de vignettes affichées par page (défaut 100)")
	parser.add_argument("--exiftool", default = "exiftool", help = "🔧 Chemin ou nom de l'exécutable ExifTool")
	parser.add_argument("--threshold", type = float, default = 0.40, help = "🎚️ Seuil de similarité cosinus (0.20 à 0.70, défaut 0.40)")
	parser.add_argument("--max-image-size", type = int, default = 2000, help = "📐 Taille maximale (pixels) des photos pour la détection")
	parser.add_argument("--min-face-size", type = int, default = 40, help = "🙂 Taille minimale (pixels) d'un visage retenu")
	parser.add_argument("--detection-score", type = float, default = 0.80, help = "🎯 Score minimal de détection d'un visage (0 à 1)")
	parser.add_argument("--no-backup", action = "store_true", help = "⚠️ Ne pas conserver de copie *_original lors de l'écriture")
	parser.add_argument("--no-emoji", action = "store_true", help = "🚫 Désactive les emojis (si Tk plante à leur affichage)")
	return parser.parse_args()


def main() -> None:
	options = parse_arguments()
	SETTINGS["emoji"] = not options.no_emoji
	# Évite les erreurs d'encodage des emojis dans les consoles Windows
	for stream in (sys.stdout, sys.stderr):
		if hasattr(stream, "reconfigure"):
			stream.reconfigure(encoding = "utf-8", errors = "replace")
	options.threshold = min(max(options.threshold, 0.20), 0.70)
	if not options.database:
		options.database = str(Path(options.data_directory).expanduser() / "face_tagger.sqlite")
	app = FaceTaggerApp(options)
	app.mainloop()


if __name__ == "__main__":
	main()
