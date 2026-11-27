"""Hook PyInstaller pour DeepFace : modules chargés par leur nom et métadonnées des paquets.

PyInstaller l'utilise grâce à l'option `--additional-hooks-dir hooks` (workflow
GitHub Actions) ou à `hookspath = ["hooks"]` (`face_tagger.spec` local). Il
remplace les options `--collect-submodules deepface.models`,
`--collect-submodules deepface.modules` et `--recursive-copy-metadata deepface`.

Pourquoi ne pas garder `--recursive-copy-metadata deepface` : DeepFace déclare
une dépendance à `onnxruntime`. Sous Windows x86_64, ce paquet est remplacé par
`onnxruntime-directml` (même module `onnxruntime`, autre nom de distribution) :
les métadonnées de `onnxruntime` n'existent plus, et la copie récursive de
PyInstaller s'arrête sur `PackageNotFoundError`. Ce hook parcourt les mêmes
dépendances, mais ignore celles qui ne sont pas installées.

Example:
	pyinstaller --onefile --additional-hooks-dir hooks face_tagger.py
"""

from __future__ import annotations

import logging
from importlib import metadata

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from PyInstaller import log as pyinstaller_log
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

logger: logging.Logger = pyinstaller_log.getLogger(__name__)


def installed_dependencies(root: str) -> list[str]:
	"""Liste une distribution et, récursivement, ses dépendances installées.

	Seules les dépendances obligatoires sont suivies : les marqueurs sont
	évalués pour la plateforme courante, sans « extra » (même règle que
	`copy_metadata(…, recursive = True)` de PyInstaller). Une dépendance
	absente est signalée puis ignorée au lieu d'arrêter la compilation.

	Args:
		root (str): Nom de la distribution de départ (`deepface`).

	Returns:
		list[str]: Noms des distributions installées, la distribution de départ en tête.
	"""
	found: list[str] = []
	seen: set[str] = set()
	pending: list[str] = [root]
	while pending:
		name: str = pending.pop()
		key: str = canonicalize_name(name)
		if key in seen:
			continue
		seen.add(key)
		try:
			distribution: metadata.Distribution = metadata.distribution(name)
		except metadata.PackageNotFoundError:
			# Cas attendu : « onnxruntime » remplacé par « onnxruntime-directml ».
			logger.warning("⚠️ Dépendance « %s » non installée : métadonnées ignorées", name)
			continue
		found.append(name)
		for line in distribution.requires or []:
			try:
				requirement: Requirement = Requirement(line)
			except InvalidRequirement:
				continue
			# Les dépendances optionnelles (extra) et celles d'une autre plateforme ne sont pas suivies
			if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
				pending.append(requirement.name)
	return found


# DeepFace importe ses détecteurs et ses modèles par leur nom (importlib) :
# PyInstaller ne peut pas les découvrir en lisant le code.
hiddenimports: list[str] = collect_submodules("deepface.models") + collect_submodules("deepface.modules")

# Métadonnées lues à l'exécution : gdown et tqdm demandent leur propre version
# (« No package metadata was found for gdown » sans elles).
datas: list[tuple[str, str]] = []
for dependency in installed_dependencies("deepface"):
	datas += copy_metadata(dependency)
