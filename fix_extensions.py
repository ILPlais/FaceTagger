#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renomme les photos de la base de Face Tagger dont l'extension ne correspond pas à leur format réel.

Une image PNG nommée « photo.jpg » s'analyse normalement (Pillow reconnaît le format d'après le contenu du fichier), mais ExifTool refuse d'y écrire les noms (« Not a valid JPG (looks more like a PNG) ») : Face Tagger l'entoure alors d'un cadre rouge. Ce script parcourt les photos indexées dans la base SQLite, reconnaît le format de chacune avec la même règle que l'application (`face_tagger.photo_real_format`), puis :
	- renomme le fichier avec l'extension de son format (« photo.jpg » → « photo.png »), en gardant la casse de l'ancienne extension (« .JPG » → « .PNG ») ;
	- renomme aussi sa sauvegarde « photo.jpg_original » (ExifTool), s'il y en a une ;
	- met à jour la base : chemin de la photo, de ses visages, de son historique et de sa sauvegarde. Les visages et étiquettes déjà mémorisés restent valables : rien ne sera ré-analysé.

Si le nouveau nom est déjà pris, un numéro est ajouté (« photo (2).png ») : rien n'est jamais écrasé. Fermez Face Tagger avant de lancer le script, car l'application garde en mémoire les anciens chemins.

Examples:
	python fix_extensions.py --dry-run
	python fix_extensions.py --directory "D:\\Photos"
	python fix_extensions.py --yes --no-emoji
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

# Barres de progression : tqdm est déjà installé avec DeepFace, qui s'en sert pour ses téléchargements
try:
	from tqdm import tqdm
except ImportError as error:
	sys.exit(f"❌ Dépendance manquante : {error}\n👉 Installez-la avec : pip install --requirement requirements.txt")

# Modules compagnons (même dossier). face_tagger fournit la règle de reconnaissance des formats (photo_real_format, avec le lecteur HEIC de pillow-heif s'il est installé), les extensions analysées et emoji() pour --no-emoji : son import n'ouvre aucune fenêtre et ne charge pas DeepFace.
try:
	import face_tagger
	from face_database import Database
	from restore_originals import BACKUP_SUFFIX
except ImportError as error:
	sys.exit(f"❌ Fichier manquant : {error}\n👉 Placez fix_extensions.py à côté de face_tagger.py, face_database.py et restore_originals.py.")

T = TypeVar("T")    # Type des éléments parcourus par une barre de progression (chemins, renommages)


@dataclass
class Rename:
	"""Renommage prévu pour une photo dont l'extension ne correspond pas à son format.

	Attributes:
		source (str): Chemin actuel de la photo, tel qu'il est enregistré dans la base.
		target (str): Nouveau chemin, avec l'extension du format réel.
		image_format (str): Format réel de la photo, selon Pillow (« `PNG` », « `JPEG` »…).
		backup_source (Optional[str]): Sauvegarde `*_original` à renommer avec la photo, ou `None` s'il n'y en a pas.
		backup_target (Optional[str]): Nouveau chemin de cette sauvegarde, ou `None` s'il n'y en a pas.
	"""
	source: str
	target: str
	image_format: str
	backup_source: Optional[str] = None
	backup_target: Optional[str] = None


def say(text: str) -> None:
	"""Affiche un message dans la console, sans ses emojis avec l'option `--no-emoji`.

	Le message passe par `tqdm.write` : il s'affiche au-dessus de la barre de progression en cours au lieu de la couper en deux (sans barre, c'est un simple `print`).

	Args:
		text (str): Message à afficher, avec ses emojis.
	"""
	tqdm.write(face_tagger.emoji(text))


def progress(items: list[T], description: str) -> Iterable[T]:
	"""Parcourt une liste en affichant une barre de progression tqdm (nombre de photos, vitesse, temps restant).

	La barre est désactivée quand la sortie n'est pas un terminal (redirection vers un fichier, test automatisé) : elle n'y laisserait que des caractères de contrôle.

	Args:
		items (list[T]): Éléments à parcourir.
		description (str): Libellé affiché devant la barre, avec ses emojis (retirés avec `--no-emoji`).

	Returns:
		Iterable[T]: Les mêmes éléments, dans le même ordre, la barre avançant à chacun.
	"""
	return tqdm(items, desc = face_tagger.emoji(description), unit = " photo", disable = None, dynamic_ncols = True)


def comparable(path: str) -> str:
	"""Calcule la forme d'un chemin qui sert à le comparer à d'autres.

	Sous Windows, `os.path.normcase` met tout en minuscules et uniformise les séparateurs : deux chemins qui ne diffèrent que par la casse désignent le même fichier.

	Args:
		path (str): Chemin à comparer.

	Returns:
		str: Chemin normalisé.
	"""
	return os.path.normcase(os.path.normpath(path))


def is_inside(path: str, directory: str) -> bool:
	"""Indique si un chemin se trouve dans un dossier (sous-dossiers compris).

	Args:
		path (str): Chemin à tester.
		directory (str): Dossier de référence (chemin absolu).

	Returns:
		bool: `True` si `path` est dans `directory` ou l'un de ses sous-dossiers.
	"""
	try:
		return os.path.commonpath([comparable(path), comparable(directory)]) == comparable(directory)
	except ValueError:
		# Sous Windows, deux lecteurs différents (C: et D:) n'ont aucun dossier commun
		return False


def matching_case(extension: str, model: str) -> str:
	"""Écrit une extension avec la casse d'une autre : en majuscules si l'ancienne l'était.

	Les appareils photo nomment souvent leurs fichiers « IMG_0001.JPG » : la photo renommée garde ainsi l'allure des autres.

	Args:
		extension (str): Nouvelle extension, en minuscules (« `.png` »).
		model (str): Ancienne extension (« `.JPG` »), éventuellement vide.

	Returns:
		str: `extension` en majuscules si les lettres de `model` le sont toutes, telle quelle sinon.
	"""
	return extension.upper() if model[1:].isupper() else extension


def free_target(
		folder: str, 
		stem: str, 
		extension: str, 
		taken: set[str]
	) -> str:
	"""Trouve un nom libre pour une photo renommée : « photo.png », sinon « photo (2).png », « photo (3).png »…

	Un nom est libre si ni le fichier, ni sa sauvegarde `*_original` n'existent sur le disque, et si aucune photo de la base (ou déjà prévue dans ce passage) ne le porte : une photo supprimée du disque peut encore y figurer.

	Args:
		folder (str): Dossier de la photo.
		stem (str): Nom de la photo sans son extension.
		extension (str): Nouvelle extension, point compris.
		taken (set[str]): Chemins déjà pris, sous la forme de `comparable`.

	Returns:
		str: Chemin libre dans `folder`.
	"""
	number = 1
	while True:
		name = f"{stem}{extension}" if number == 1 else f"{stem} ({number}){extension}"
		candidate = os.path.join(folder, name)
		if (
			comparable(candidate) not in taken
			and not os.path.lexists(candidate)
			and not os.path.lexists(candidate + BACKUP_SUFFIX)
		):
			return candidate
		number += 1


def plan_renames(
		paths: Iterable[str], 
		taken: set[str]
	) -> tuple[list[Rename], int]:
	"""Contrôle le format des photos et prévoit le renommage de celles dont l'extension ne convient pas.

	Seul l'en-tête de chaque fichier est lu : le contrôle est rapide, même sur une grande photothèque. Les photos absentes du disque sont comptées et ignorées ; celles dont Pillow ne reconnaît pas le format, ignorées sans message (Face Tagger n'a pas pu les analyser non plus).

	Args:
		paths (Iterable[str]): Chemins des photos à contrôler (clés de la base).
		taken (set[str]): Chemins déjà pris (sous la forme de `comparable`), complété au fur et à mesure par les nouveaux noms prévus.

	Returns:
		tuple[list[Rename], int]: Renommages prévus, et nombre de photos introuvables sur le disque.
	"""
	paths = list(paths)
	renames: list[Rename] = []
	missing = 0
	for path in progress(paths, "🔍 Contrôle"):
		if not os.path.isfile(path):
			missing += 1
			continue
		real_format = face_tagger.photo_real_format(path)
		if real_format is None:
			continue
		image_format, expected = real_format
		folder, name = os.path.split(path)
		stem, extension = os.path.splitext(name)
		if extension.lower() in expected:
			continue
		target = free_target(folder, stem, matching_case(expected[0], extension), taken)
		taken.add(comparable(target))
		backup = path + BACKUP_SUFFIX
		has_backup = os.path.isfile(backup)
		renames.append(Rename(
			source = path,
			target = target,
			image_format = image_format,
			backup_source = backup if has_backup else None,
			backup_target = target + BACKUP_SUFFIX if has_backup else None,
		))
	return renames, missing


def apply_renames(
		renames: list[Rename], 
		database: Database, 
		log: Callable[[str], None] = say
	) -> tuple[int, int]:
	"""Renomme les photos (et leurs sauvegardes `*_original`), puis reporte chaque renommage dans la base.

	Un échec sur une photo n'interrompt pas les suivantes. Si la base refuse un renommage, les fichiers reprennent leur ancien nom : fichiers et base restent ainsi d'accord.

	Args:
		renames (list[Rename]): Renommages prévus par `plan_renames`.
		database (Database): Base de Face Tagger à mettre à jour.
		log (Callable[[str], None]): Fonction qui reçoit chaque message (`say` par défaut).

	Returns:
		tuple[int, int]: Nombre de photos renommées et nombre d'échecs.
	"""
	done = failed = 0
	for item in progress(renames, "📝 Renommage"):
		# Sous Linux et macOS, os.rename écraserait sans prévenir un fichier apparu depuis la préparation
		if os.path.lexists(item.target):
			failed += 1
			log(f"❌ {item.source} : « {os.path.basename(item.target)} » existe déjà, photo laissée telle quelle.")
			continue
		try:
			os.rename(item.source, item.target)
		except OSError as error:
			failed += 1
			log(f"❌ Échec pour {item.source} : {error}")
			continue

		# La sauvegarde suit la photo, sinon restore_originals ne saurait plus à quelle photo elle correspond
		backup_moved = False
		if item.backup_source and item.backup_target:
			try:
				if os.path.lexists(item.backup_target):
					raise FileExistsError(f"« {os.path.basename(item.backup_target)} » existe déjà")
				os.rename(item.backup_source, item.backup_target)
				backup_moved = True
			except OSError as error:
				log(f"⚠️ Sauvegarde non renommée ({item.backup_source}) : {error}")

		try:
			database.rename_photo(
				item.source,
				item.target,
				item.backup_source if backup_moved else None,
				item.backup_target if backup_moved else None,
			)
		except Exception as error:
			failed += 1
			log(f"❌ Base non mise à jour pour {item.source} : {error}")
			undo_rename(item, backup_moved, log)
			continue

		done += 1
		log(f"✅ {item.source} renommée en « {os.path.basename(item.target)} »" + (" (sauvegarde comprise)" if backup_moved else ""))
	return done, failed


def undo_rename(
		item: Rename, 
		backup_moved: bool, 
		log: Callable[[str], None]
	) -> None:
	"""Redonne à une photo (et à sa sauvegarde) son ancien nom, quand la base n'a pas pu suivre le renommage.

	Args:
		item (Rename): Renommage à annuler.
		backup_moved (bool): `True` si la sauvegarde `*_original` avait été renommée elle aussi.
		log (Callable[[str], None]): Fonction qui reçoit le message si l'annulation échoue.
	"""
	try:
		if backup_moved and item.backup_source and item.backup_target:
			os.rename(item.backup_target, item.backup_source)
		os.rename(item.target, item.source)
	except OSError as error:
		log(f"⚠️ Impossible de redonner son nom à {item.source} : {error}. Renommez-la à la main.")


def ask_confirmation(message: str) -> bool:
	"""Demande une confirmation dans la console.

	Args:
		message (str): Question posée.

	Returns:
		bool: `True` si l'utilisateur répond « o », « oui », « y » ou « yes » ; `False` sinon, ou si la console ne permet pas de répondre.
	"""
	try:
		answer = input(face_tagger.emoji(f"❓ {message} [o/N] "))
	except EOFError:
		return False
	return answer.strip().lower() in ("o", "oui", "y", "yes")


def parse_arguments() -> argparse.Namespace:
	"""Lit les options de la ligne de commande.

	Le chemin par défaut de la base dépend du dossier de données : il est complété par `main()`.

	Returns:
		argparse.Namespace: Options lues : `directory`, `data_directory`, `database`, `dry_run`, `yes` et `no_emoji`.
	"""
	parser = argparse.ArgumentParser(
		description = "🏷️ Renomme les photos de la base de Face Tagger dont l'extension ne correspond pas au format réel."
	)
	parser.add_argument(
		"--directory",
		default = "",
		help = "📁 Ne traite que les photos de ce dossier (sous-dossiers compris) ; toute la base par défaut"
	)
	parser.add_argument(
		"--data-directory",
		default = str(Path.home() / ".face_tagger"),
		help = "📂 Dossier de données de Face Tagger (défaut : ~/.face_tagger)"
	)
	parser.add_argument(
		"--database",
		default = "",
		help = "🗄️ Base SQLite de Face Tagger (défaut : <data-directory>/face_tagger.sqlite)"
	)
	parser.add_argument(
		"--dry-run",
		action = "store_true",
		help = "🔍 Affiche les renommages prévus, sans rien modifier"
	)
	parser.add_argument(
		"--yes",
		action = "store_true",
		help = "✅ Ne demande pas de confirmation"
	)
	parser.add_argument(
		"--no-emoji",
		action = "store_true",
		help = "🚫 Messages sans emojis"
	)
	return parser.parse_args()


def main() -> int:
	"""Point d'entrée en ligne de commande : contrôle les photos de la base, demande confirmation, puis les renomme.

	Returns:
		int: Code de sortie du programme : `0` en cas de succès, de simulation, d'annulation ou s'il n'y a rien à renommer ; `1` si la base ou le dossier sont introuvables, ou si au moins une photo n'a pas pu être renommée.
	"""
	options = parse_arguments()
	face_tagger.SETTINGS["emoji"] = not options.no_emoji
	# Évite les erreurs d'encodage des emojis et des accents dans les consoles Windows
	for stream in (sys.stdout, sys.stderr):
		if hasattr(stream, "reconfigure"):
			stream.reconfigure(encoding = "utf-8", errors = "replace")

	database_path = Path(options.database or Path(options.data_directory) / "face_tagger.sqlite").expanduser()
	if not database_path.is_file():
		say(f"❌ Base introuvable : {database_path}")
		return 1
	directory = ""
	if options.directory:
		directory = os.path.abspath(os.path.expanduser(options.directory))
		if not os.path.isdir(directory):
			say(f"❌ Dossier introuvable : {directory}")
			return 1

	try:
		database = Database(database_path)
	except Exception as error:
		say(f"❌ Base illisible ({database_path}) : {error}")
		return 1

	known = sorted(database.signatures())
	paths = [path for path in known if not directory or is_inside(path, directory)]
	where = f" de {directory}" if directory else ""
	say(f"🔍 Contrôle du format de {len(paths)} photo{'s' if len(paths) > 1 else ''}{where} indexée{'s' if len(paths) > 1 else ''} dans {database_path}…")
	renames, missing = plan_renames(paths, {comparable(path) for path in known})
	if missing:
		say(f"⚠️ {missing} photo{'s' if missing != 1 else ''} de la base introuvable{'s' if missing > 1 else ''} sur le disque (ignorée{'s' if missing > 1 else ''}).")
	if not renames:
		say("✅ Toutes les extensions correspondent au format des photos : rien à renommer.")
		return 0

	say(f"📋 {len(renames)} photo{'s' if len(renames) > 1 else ''} à renommer :")
	for item in renames:
		say(f"  ✏️ {item.source} en « {os.path.basename(item.target)} » (image {item.image_format})")
		if item.backup_source and item.backup_target:
			say(f"     💾 et sa sauvegarde en « {os.path.basename(item.backup_target)} »")
	unsupported = [item for item in renames if Path(item.target).suffix.lower() not in face_tagger.IMAGE_EXTENSIONS]
	if unsupported:
		say(f"⚠️ {len(unsupported)} photo{'s' if len(unsupported) > 1 else ''} prendront une extension que Face Tagger n'analyse pas : elles n'apparaîtront plus dans l'application.")

	if options.dry_run:
		say("🔍 Simulation : aucun fichier n'a été renommé.")
		return 0
	if not options.yes:
		say("⚠️ Fermez Face Tagger avant de continuer : il garde en mémoire les anciens noms des photos.")
		if not ask_confirmation(f"Renommer {len(renames)} photo{'s' if len(renames) > 1 else ''} ?"):
			say("⏹️ Opération annulée.")
			return 0

	done, failed = apply_renames(renames, database)
	say(f"🎉 Terminé : {done} photo{'s' if done != 1 else ''} renommée{'s' if done > 1 else ''}, {failed} échec{'s' if failed > 1 else ''}.")
	if done:
		say("👉 Relancez l'analyse dans Face Tagger : les photos renommées y gardent leurs visages, sans nouvelle analyse.")
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
