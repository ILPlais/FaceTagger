#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_database.py — Base SQLite de Face Tagger (bibliothèque standard uniquement).

Elle mémorise, pour chaque photo :
	- sa « signature » (date de modification + taille) : si elle change, tout est recalculé ;
	- les visages détectés (position, empreinte numérique, vignette JPEG) ;
	- les étiquettes lues par ExifTool (régions, PersonInImage, mots-clés) ;
	- l'historique des modifications faites par l'application (noms écrits, vérification,
	fichier de sauvegarde *_original) et leur éventuelle restauration.

Les analyses et lectures d'étiquettes déjà faites ne sont ainsi jamais refaites tant que la photo
n'a pas changé. Chaque thread utilise sa propre connexion (SQLite en mode WAL).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_VERSION = 1
THUMBNAIL_CACHE_SIZE = 3000   # Nombre de vignettes JPEG gardées en mémoire (cache LRU)
QUERY_CHUNK = 500             # Taille des lots pour les requêtes « IN (…) »

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
	key TEXT PRIMARY KEY,
	value TEXT NOT NULL
);

-- Une ligne par photo : signature du fichier et étiquettes lues (JSON brut d'ExifTool)
CREATE TABLE IF NOT EXISTS photos (
	path TEXT PRIMARY KEY,
	mtime_ns INTEGER NOT NULL,
	size INTEGER NOT NULL,
	analysed INTEGER NOT NULL DEFAULT 0,
	tags_json TEXT
);

-- Un visage détecté : position normalisée, empreinte (128 flottants) et vignette JPEG
CREATE TABLE IF NOT EXISTS faces (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	photo TEXT NOT NULL,
	x REAL NOT NULL,
	y REAL NOT NULL,
	w REAL NOT NULL,
	h REAL NOT NULL,
	score REAL NOT NULL,
	embedding BLOB NOT NULL,
	thumbnail BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_photo ON faces(photo);

-- Historique des écritures de métadonnées
CREATE TABLE IF NOT EXISTS modifications (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	photo TEXT NOT NULL,
	written_at TEXT NOT NULL,
	names TEXT NOT NULL,
	fields TEXT NOT NULL,
	verified INTEGER NOT NULL,
	backup_path TEXT,
	restored_at TEXT
);
CREATE INDEX IF NOT EXISTS modifications_photo ON modifications(photo);
"""


def utc_now() -> str:
	"""Date et heure UTC au format ISO 8601, à la seconde."""
	return datetime.now(timezone.utc).isoformat(timespec = "seconds")


class Database:
	"""Accès à la base SQLite de Face Tagger (utilisable depuis plusieurs threads)."""

	def __init__(self, path: Path | str):
		self.path = Path(path)
		self.path.parent.mkdir(parents = True, exist_ok = True)
		self._local = threading.local()                       # Une connexion par thread
		self._thumbnails: OrderedDict[int, bytes] = OrderedDict()
		self._thumbnails_lock = threading.Lock()
		self.connection.executescript(SCHEMA)
		version = self.get_meta("schema_version")
		if version is None:
			self.set_meta("schema_version", str(SCHEMA_VERSION))
		elif int(version) > SCHEMA_VERSION:
			raise RuntimeError(f"La base {self.path} a été créée par une version plus récente de Face Tagger.")

	# ---------- Connexion ----------

	@property
	def connection(self) -> sqlite3.Connection:
		"""Connexion du thread courant (créée à la demande)."""
		connection = getattr(self._local, "connection", None)
		if connection is None:
			connection = sqlite3.connect(str(self.path), timeout = 30)
			connection.execute("PRAGMA journal_mode = WAL")      # Lecteurs et écrivain ne se bloquent pas
			connection.execute("PRAGMA synchronous = NORMAL")
			self._local.connection = connection
		return connection

	def close(self) -> None:
		"""Ferme la connexion du thread courant."""
		connection = getattr(self._local, "connection", None)
		if connection is not None:
			connection.close()
			self._local.connection = None

	# ---------- Métadonnées de la base ----------

	def get_meta(self, key: str) -> Optional[str]:
		row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
		return row[0] if row else None

	def set_meta(self, key: str, value: str) -> None:
		with self.connection as connection:
			connection.execute(
				"INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
				(key, value),
			)

	def reset_analysis_if_needed(self, signature: str) -> bool:
		"""Invalide les visages mémorisés si les paramètres de détection ont changé.
		Les étiquettes lues, elles, restent valables. Retourne True si une invalidation a eu lieu."""
		previous = self.get_meta("analysis_signature")
		if previous == signature:
			return False
		with self.connection as connection:
			connection.execute("DELETE FROM faces")
			connection.execute("UPDATE photos SET analysed = 0")
			connection.execute(
				"INSERT INTO meta(key, value) VALUES('analysis_signature', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
				(signature,),
			)
		with self._thumbnails_lock:
			self._thumbnails.clear()
		return previous is not None

	# ---------- Photos et visages ----------

	def signatures(self) -> dict[str, tuple[int, int, bool]]:
		"""Retourne {chemin: (date de modification en ns, taille, visages déjà analysés)}."""
		rows = self.connection.execute("SELECT path, mtime_ns, size, analysed FROM photos")
		return {path: (mtime_ns, size, bool(analysed)) for path, mtime_ns, size, analysed in rows}

	def store_analysis(self, path: str, mtime_ns: int, size: int, faces: list[dict]) -> None:
		"""Enregistre les visages d'une photo (en remplaçant les anciens). Chaque visage est un dictionnaire :
		box (x, y, l, h), score, embedding (octets) et thumbnail (octets JPEG).
		Les étiquettes déjà lues sont conservées si la signature du fichier n'a pas changé."""
		with self.connection as connection:
			connection.execute("DELETE FROM faces WHERE photo = ?", (path,))
			connection.execute(
				"""
				INSERT INTO photos(path, mtime_ns, size, analysed) VALUES(?, ?, ?, 1)
				ON CONFLICT(path) DO UPDATE SET
					tags_json = CASE WHEN photos.mtime_ns = excluded.mtime_ns AND photos.size = excluded.size
						THEN photos.tags_json ELSE NULL END,
					mtime_ns = excluded.mtime_ns,
					size = excluded.size,
					analysed = 1
				""",
				(path, mtime_ns, size),
			)
			connection.executemany(
				"INSERT INTO faces(photo, x, y, w, h, score, embedding, thumbnail) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
				[(path, *face["box"], face["score"], face["embedding"], face["thumbnail"]) for face in faces],
			)

	def load_faces(self, photos: set[str]) -> list[tuple]:
		"""Retourne les visages des photos données : (id, photo, x, y, l, h, score, empreinte).
		Les vignettes ne sont PAS chargées ici : elles sont lues à la demande (voir `thumbnails`)."""
		rows = self.connection.execute("SELECT id, photo, x, y, w, h, score, embedding FROM faces ORDER BY id")
		return [row for row in rows if row[1] in photos]

	def thumbnails(self, face_ids: Iterable[int]) -> dict[int, bytes]:
		"""Retourne les vignettes JPEG des visages demandés, avec un cache LRU en mémoire."""
		result: dict[int, bytes] = {}
		missing: list[int] = []
		with self._thumbnails_lock:
			for face_id in dict.fromkeys(face_ids):
				data = self._thumbnails.get(face_id)
				if data is None:
					missing.append(face_id)
				else:
					self._thumbnails.move_to_end(face_id)
					result[face_id] = data
		for start in range(0, len(missing), QUERY_CHUNK):
			chunk = missing[start:start + QUERY_CHUNK]
			marks = ",".join("?" * len(chunk))
			rows = self.connection.execute(f"SELECT id, thumbnail FROM faces WHERE id IN ({marks})", chunk).fetchall()
			with self._thumbnails_lock:
				for face_id, data in rows:
					result[face_id] = data
					self._thumbnails[face_id] = data
				while len(self._thumbnails) > THUMBNAIL_CACHE_SIZE:
					self._thumbnails.popitem(last = False)
		return result

	# ---------- Étiquettes lues dans les photos ----------

	def tags(self, photos: Iterable[str]) -> dict[str, dict]:
		"""Retourne les étiquettes mémorisées {chemin: JSON ExifTool} (uniquement celles déjà lues)."""
		photos = list(photos)
		result: dict[str, dict] = {}
		for start in range(0, len(photos), QUERY_CHUNK):
			chunk = photos[start:start + QUERY_CHUNK]
			marks = ",".join("?" * len(chunk))
			rows = self.connection.execute(
				f"SELECT path, tags_json FROM photos WHERE tags_json IS NOT NULL AND path IN ({marks})", chunk,
			)
			for path, text in rows:
				result[path] = json.loads(text)
		return result

	def store_tags(self, entries: dict[str, dict]) -> None:
		"""Mémorise les étiquettes lues {chemin: JSON ExifTool} pour des photos déjà connues."""
		with self.connection as connection:
			connection.executemany(
				"UPDATE photos SET tags_json = ? WHERE path = ?",
				[(json.dumps(entry, ensure_ascii = False), path) for path, entry in entries.items()],
			)

	def update_signature(self, path: str, mtime_ns: int, size: int) -> None:
		"""Met à jour la signature d'une photo que l'application vient de modifier elle-même.
		Les pixels n'ayant pas changé, les visages mémorisés restent valables."""
		with self.connection as connection:
			connection.execute("UPDATE photos SET mtime_ns = ?, size = ? WHERE path = ?", (mtime_ns, size, path))

	# ---------- Historique des modifications ----------

	def record_modification(self, photo: str, names: list[str], fields: list[str], verified: bool,
							backup_path: Optional[str]) -> None:
		"""Garde une trace d'une écriture de métadonnées (et du fichier de sauvegarde associé)."""
		with self.connection as connection:
			connection.execute(
				"""
				INSERT INTO modifications(photo, written_at, names, fields, verified, backup_path)
				VALUES(?, ?, ?, ?, ?, ?)
				""",
				(photo, utc_now(), json.dumps(names, ensure_ascii = False), json.dumps(fields), int(verified), backup_path),
			)

	def mark_restored(self, photo: str, mtime_ns: int, size: int) -> None:
		"""Enregistre qu'une photo a été restaurée depuis sa sauvegarde : ses étiquettes mémorisées sont
		oubliées (elles seront relues) mais ses visages restent valables (mêmes pixels)."""
		# Sous Windows, les chemins ne distinguent pas majuscules et minuscules
		match = "= ? COLLATE NOCASE" if os.name == "nt" else "= ?"
		with self.connection as connection:
			connection.execute(
				f"UPDATE modifications SET restored_at = ? WHERE photo {match} AND restored_at IS NULL", (utc_now(), photo),
			)
			connection.execute(
				f"UPDATE photos SET mtime_ns = ?, size = ?, tags_json = NULL WHERE path {match}", (mtime_ns, size, photo),
			)

	def mark_backup_deleted(self, backup_path: str) -> None:
		"""Enregistre qu'un fichier de sauvegarde a été supprimé."""
		match = "= ? COLLATE NOCASE" if os.name == "nt" else "= ?"
		with self.connection as connection:
			connection.execute(f"UPDATE modifications SET backup_path = NULL WHERE backup_path {match}", (backup_path,))

	def statistics(self) -> dict[str, int]:
		"""Quelques compteurs pour le journal."""
		def count(query: str) -> int:
			return self.connection.execute(query).fetchone()[0]
		return {
			"photos": count("SELECT COUNT(*) FROM photos"),
			"faces": count("SELECT COUNT(*) FROM faces"),
			"modifications": count("SELECT COUNT(*) FROM modifications"),
			"backups": count("SELECT COUNT(*) FROM modifications WHERE backup_path IS NOT NULL AND restored_at IS NULL"),
		}
