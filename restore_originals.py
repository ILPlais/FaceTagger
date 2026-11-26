#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restaure (ou supprime) d'un coup les sauvegardes *_original créées par ExifTool.

Pour chaque fichier « photo.jpg_original » trouvé dans le dossier et ses sous-dossiers :
	--action restore   (défaut) : « photo.jpg_original » remplace « photo.jpg » (état d'avant modification)
	--action delete             : « photo.jpg_original » est supprimé (on garde la photo modifiée)

Si la base SQLite de Face Tagger existe, elle est mise à jour (la photo restaurée sera relue au prochain passage). Seuls les fichiers dont le nom de base a une extension d'image sont concernés.

Examples:
	python restore_originals.py --directory D:\\Photos --dry-run
	python restore_originals.py --directory D:\\Photos
	python restore_originals.py --directory D:\\Photos --action delete --yes
	python restore_originals.py                       (ouvre un sélecteur de dossier)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
	# Pour les annotations seulement : la base est facultative, ce script reste utilisable sans elle
	from face_database import Database

# Suffixe ajouté par ExifTool aux sauvegardes, et extensions d'images concernées
BACKUP_SUFFIX: str = "_original"
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def find_backups(directory: str) -> list[tuple[str, str]]:
	"""Cherche les sauvegardes `*_original` d'un dossier et de ses sous-dossiers.

	Seules les sauvegardes de photos (extension d'image avant « `_original` ») sont retenues.

	Args:
		directory (str): Dossier à parcourir.

	Returns:
		list[tuple[str, str]]: Couples (chemin de la sauvegarde, chemin de la photo correspondante), triés par dossier puis par nom.
	"""
	backups: list[tuple[str, str]] = []
	for root, directories, files in os.walk(directory):
		directories.sort()
		for file_name in sorted(files):
			if not file_name.endswith(BACKUP_SUFFIX):
				continue
			target_name = file_name[:-len(BACKUP_SUFFIX)]
			# On ignore les fichiers qui ne sont pas des sauvegardes de photos
			if Path(target_name).suffix.lower() not in IMAGE_EXTENSIONS:
				continue
			backups.append((os.path.join(root, file_name), os.path.join(root, target_name)))
	return backups


def restore_backups(
		backups: list[tuple[str, str]],
		database: Optional[Database] = None,
		log: Callable[[str], None] = print,
		dry_run: bool = False
	) -> tuple[int, int]:
	"""Remplace chaque photo par sa sauvegarde `*_original`, qui disparaît alors.

	Un échec sur une photo n'interrompt pas les suivantes. La base, si elle est fournie, est mise à jour photo par photo.

	Args:
		backups (list[tuple[str, str]]): Couples (sauvegarde, photo), comme ceux renvoyés par `find_backups`.
		database (Optional[Database]): Base de Face Tagger à mettre à jour, ou `None` pour ne pas y toucher.
		log (Callable[[str], None]): Fonction qui reçoit chaque message (`print` par défaut).
		dry_run (bool): `True` pour seulement annoncer ce qui serait fait, sans rien modifier.

	Returns:
		tuple[int, int]: Nombre de photos restaurées (ou qui le seraient, en simulation) et nombre d'échecs.
	"""
	done = failed = 0
	for backup, target in backups:
		try:
			existed = os.path.exists(target)
			if dry_run:
				log(f"🔍 [simulation] {backup}  →  {target}")
			else:
				# os.replace écrase la photo modifiée, y compris sous Windows
				os.replace(backup, target)
				log(f"♻️ Restauré : {target}" + ("" if existed else " (fichier recréé)"))
				update_database(database, "restore", backup, target, log)
			done += 1
		except OSError as error:
			failed += 1
			log(f"❌ Échec pour {target} : {error}")
	return done, failed


def delete_backups(backups: list[tuple[str, str]], database: Optional[Database] = None, log: Callable[[str], None] = print,
				dry_run: bool = False) -> tuple[int, int]:
	"""Supprime les fichiers de sauvegarde *_original ; les photos modifiées sont conservées.

	Un échec sur une sauvegarde n'interrompt pas les suivantes. La base, si elle est fournie, est mise à jour
	sauvegarde par sauvegarde.

	Args:
		backups (list[tuple[str, str]]): Couples (sauvegarde, photo), comme ceux renvoyés par `find_backups`.
		database (Optional[Database]): Base de Face Tagger à mettre à jour, ou None pour ne pas y toucher.
		log (Callable[[str], None]): Fonction qui reçoit chaque message (`print` par défaut).
		dry_run (bool): `True` pour seulement annoncer ce qui serait fait, sans rien supprimer.

	Returns:
		tuple[int, int]: Nombre de sauvegardes supprimées (ou qui le seraient, en simulation) et nombre d'échecs.
	"""
	done = failed = 0
	for backup, target in backups:
		try:
			if dry_run:
				log(f"🔍 [simulation] suppression de {backup}")
			else:
				os.remove(backup)
				log(f"🗑️ Supprimé : {backup}")
				update_database(database, "delete", backup, target, log)
			done += 1
		except OSError as error:
			failed += 1
			log(f"❌ Échec pour {backup} : {error}")
	return done, failed


def update_database(
		database: Optional[Database],
		action: str,
		backup: str,
		target: str,
		log: Callable[[str], None]
	) -> None:
	"""Répercute une restauration ou une suppression dans la base SQLite.

	Une erreur de base est seulement signalée : elle n'interrompt jamais l'opération sur les fichiers.

	Args:
		database (Optional[Database]): Base à mettre à jour ; `None` : il n'y a rien à faire.
		action (str): « `restore` » (photo restaurée) ou « `delete` » (sauvegarde supprimée).
		backup (str): Chemin de la sauvegarde `*_original`.
		target (str): Chemin de la photo.
		log (Callable[[str], None]): Fonction qui reçoit le message d'avertissement en cas d'erreur.
	"""
	if database is None:
		return
	try:
		if action == "restore":
			stat = os.stat(target)
			database.mark_restored(target, stat.st_mtime_ns, stat.st_size)
		else:
			database.mark_backup_deleted(backup)
	except Exception as error:
		log(f"⚠️ Base de données non mise à jour pour {target} : {error}")


def ask_confirmation(message: str, graphical: bool) -> bool:
	"""Demande une confirmation à l'utilisateur.

	Args:
		message (str): Question posée.
		graphical (bool): `True` pour une boîte de dialogue (lancement sans `--directory`), False pour la console.

	Returns:
		bool: `True` si l'utilisateur accepte (« o », « oui », « y » ou « yes » dans la console).
	"""
	if graphical:
		from tkinter import messagebox
		return messagebox.askyesno("❓ Confirmation", message)
	return input(f"❓ {message} [o/N] ").strip().lower() in ("o", "oui", "y", "yes")


def choose_directory() -> str:
	"""Ouvre un sélecteur de dossier graphique (utilisé quand `--directory` n'est pas fourni).

	Returns:
		str: Dossier choisi, ou chaîne vide si l'utilisateur a annulé.
	"""
	import tkinter as tk
	from tkinter import filedialog
	root = tk.Tk()
	root.withdraw()
	directory = filedialog.askdirectory(title = "📁 Dossier contenant les fichiers *_original")
	root.destroy()
	return directory


def parse_arguments() -> argparse.Namespace:
	"""Lit les options de la ligne de commande.

	Returns:
		argparse.Namespace: Options lues : `directory`, `action`, `dry_run`, `yes` et `database`.
	"""
	parser = argparse.ArgumentParser(description = "♻️ Restaure ou supprime d'un coup les sauvegardes *_original créées par ExifTool.")
	parser.add_argument(
		"--directory",
		default = "",
		help = "📁 Dossier à parcourir (sous-dossiers compris) ; sélecteur graphique si absent"
	)
	parser.add_argument(
		"--action",
		choices = ("restore", "delete"),
		default = "restore",
		help = "♻️ restore  : remet les photos d'origine (défaut) — 🗑️ delete : supprime les sauvegardes"
	)
	parser.add_argument(
		"--dry-run",
		action = "store_true",
		help = "🔍 Affiche ce qui serait fait, sans rien modifier"
	)
	parser.add_argument(
		"--yes",
		action = "store_true",
		help = "✅ Ne demande pas de confirmation"
	)
	parser.add_argument(
		"--database",
		default = str(Path.home() / ".face_tagger" / "face_tagger.sqlite"),
		help = "🗄️ Base SQLite de Face Tagger à mettre à jour (ignorée si elle n'existe pas)"
	)
	return parser.parse_args()


def main() -> int:
	"""Point d'entrée en ligne de commande : cherche les sauvegardes, demande confirmation, puis restaure ou supprime.

	Returns:
		int: Code de sortie du programme : `0` en cas de succès, d'annulation ou d'absence de sauvegarde ; 1 si aucun dossier valide n'a été choisi ou si au moins un fichier a échoué.
	"""
	options = parse_arguments()
	# Évite les erreurs d'encodage des emojis dans les consoles Windows
	for stream in (sys.stdout, sys.stderr):
		if hasattr(stream, "reconfigure"):
			stream.reconfigure(encoding = "utf-8", errors = "replace")

	graphical = not options.directory
	directory = options.directory or choose_directory()
	if not directory:
		print("❌ Aucun dossier choisi.")
		return 1
	directory = os.path.abspath(os.path.expanduser(directory))
	if not os.path.isdir(directory):
		print(f"❌ Dossier introuvable : {directory}")
		return 1

	print(f"🔍 Recherche des fichiers *{BACKUP_SUFFIX} dans {directory}…")
	backups = find_backups(directory)
	if not backups:
		print("ℹ️ Aucune sauvegarde trouvée.")
		return 0
	print(f"📦 {len(backups)} sauvegarde(s) trouvée(s).")

	if not options.dry_run and not options.yes:
		if options.action == "restore":
			message = f"Remplacer {len(backups)} photo(s) par leur version d'origine ? Les métadonnées actuelles seront perdues."
		else:
			message = f"Supprimer définitivement {len(backups)} fichier(s) {BACKUP_SUFFIX} ?"
		if not ask_confirmation(message, graphical):
			print("⏹️ Opération annulée.")
			return 0

	# La base n'est utilisée que si elle existe déjà et si le module est disponible
	database: Optional[Database] = None
	if Path(options.database).exists() and not options.dry_run:
		try:
			from face_database import Database
			database = Database(options.database)
		except Exception as error:
			print(f"⚠️ Base de données ignorée : {error}")

	action = restore_backups if options.action == "restore" else delete_backups
	done, failed = action(backups, database = database, dry_run = options.dry_run)
	summary = f"🎉 Terminé : {done} fichier(s) traité(s), {failed} échec(s)." + (" (simulation)" if options.dry_run else "")
	print(summary)
	if graphical:
		from tkinter import Tk, messagebox
		root = Tk()
		root.withdraw()
		messagebox.showinfo("🎉 Terminé", summary)
		root.destroy()
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
