#!/usr/bin/env python3
"""
Script pour extraire une portion d'un fichier audio
"""

import librosa
import soundfile as sf

# Charger l'audio
y, sr = librosa.load('test.wav', sr=None)

# Calculer le nombre d'échantillons pour 7 secondes
samples_7s = int(sr * 7)

# Extraire les 7 premières secondes
y_7s = y[:samples_7s]

# Sauvegarder
sf.write('test7s.wav', y_7s, sr)
print(f"Fichier test7s.wav créé avec succès") 