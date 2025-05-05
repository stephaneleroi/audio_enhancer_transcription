#!/usr/bin/env python3
"""
Programme d'amélioration audio pour les enregistrements de réunions avec des locuteurs inaudibles.
Version simplifiée optimisée pour le traitement standard qui donne les meilleurs résultats.
"""

import os
import sys
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Any, Tuple, Dict
from dotenv import load_dotenv
from transcription import transcribe_audio_improved
import json
import time

def setup_logging(verbose=False):
    """Configure le système de journalisation."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

# Ajoutons une fonction de compression dynamique pour rehausser les voix faibles tout en évitant la distorsion
def dynamic_range_compression(y, sr, threshold_db=-30, ratio=4.0, attack_ms=5.0, release_ms=50.0, logger=None):
    """
    Applique une compression dynamique pour rehausser les voix faibles sans distorsion.
    
    Args:
        y: signal audio
        sr: taux d'échantillonnage
        threshold_db: seuil de compression (dB)
        ratio: ratio de compression (ex: 4:1)
        attack_ms: temps d'attaque en ms
        release_ms: temps de relâchement en ms
    """
    if logger:
        logger.info(f"Application de compression dynamique: seuil={threshold_db}dB, ratio={ratio}:1")
    start_time = time.time()
    threshold_linear = 10 ** (threshold_db / 20.0)
    attack_samples = int(sr * attack_ms / 1000.0)
    release_samples = int(sr * release_ms / 1000.0)
    amplitude = np.abs(y)
    envelope = np.copy(amplitude)
    # Vectorisation de l'enveloppe (approximation rapide)
    alpha_a = 1.0 / max(attack_samples, 1)
    alpha_r = 1.0 / max(release_samples, 1)
    for i in range(1, len(amplitude)):
        if amplitude[i] > envelope[i-1]:
            envelope[i] = alpha_a * amplitude[i] + (1 - alpha_a) * envelope[i-1]
        else:
            envelope[i] = alpha_r * amplitude[i] + (1 - alpha_r) * envelope[i-1]
    gain = np.ones_like(envelope)
    mask = envelope < threshold_linear
    gain[mask] = (threshold_linear / np.maximum(envelope[mask], 1e-10)) ** (1.0 - 1.0/ratio)
    max_gain = 10.0
    gain = np.minimum(gain, max_gain)
    # Vectorisation du lissage du gain
    window_size = int(sr * 0.02)
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        gain_smoothed = np.convolve(gain, kernel, mode='same')
    else:
        gain_smoothed = gain
    y_compressed = y * gain_smoothed
    if logger:
        logger.info(f"Compression dynamique terminée en {time.time() - start_time:.2f} secondes.")
    return y_compressed

def spectral_subtraction(y, sr, frame_len=2048, hop_len=512, noise_reduce_factor=2.0, logger=None):
    """
    Applique une soustraction spectrale pour réduire le bruit de fond.
    """
    if logger:
        logger.info(f"Application de soustraction spectrale avec facteur={noise_reduce_factor}")
    
    # Calculer le STFT
    D = librosa.stft(y, n_fft=frame_len, hop_length=hop_len)
    
    # Obtenir les magnitudes et phases
    mag, phase = librosa.magphase(D)
    
    # Estimer le niveau de bruit (en utilisant les 5% les plus silencieux comme référence)
    percentile = 5
    noise_mag = np.percentile(mag, percentile, axis=1)
    noise_mag = noise_mag.reshape(-1, 1)
    
    # Appliquer la soustraction spectrale
    mag_filtered = np.maximum(mag - noise_mag * noise_reduce_factor, 0.01 * noise_mag)
    
    # Reconstruire le signal
    D_filtered = mag_filtered * phase
    y_filtered = librosa.istft(D_filtered, hop_length=hop_len)
    
    # Ajuster la longueur du signal filtré pour correspondre à l'original
    if len(y_filtered) < len(y):
        y_filtered = librosa.util.fix_length(y_filtered, size=len(y))
    else:
        y_filtered = y_filtered[:len(y)]
    
    return y_filtered

def multi_level_amplification(y, sr, weak_threshold_db=-25, very_weak_threshold_db=-35, 
                            weak_boost=3.0, very_weak_boost=4.5, logger=None):
    """
    Amplifie les voix faibles avec différents niveaux d'amplification selon l'intensité.
    Laisse les voix fortes complètement intactes.
    """
    if logger:
        logger.info(f"Amplification multi-niveaux: seuil faible={weak_threshold_db}dB (boost x{weak_boost}), " +
                   f"seuil très faible={very_weak_threshold_db}dB (boost x{very_weak_boost})")
    start_time = time.time()
    weak_threshold = 10 ** (weak_threshold_db / 20)
    very_weak_threshold = 10 ** (very_weak_threshold_db / 20)
    window_size = int(sr * 0.02)
    amplitude = np.abs(y)
    # Vectorisation du lissage de l'amplitude
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        smoothed_amplitude = np.convolve(amplitude, kernel, mode='same')
    else:
        smoothed_amplitude = amplitude
    normal_mask = smoothed_amplitude >= weak_threshold
    weak_mask = (smoothed_amplitude < weak_threshold) & (smoothed_amplitude >= very_weak_threshold)
    very_weak_mask = smoothed_amplitude < very_weak_threshold
    gain = np.ones_like(y)
    gain[weak_mask] = weak_boost
    gain[very_weak_mask] = very_weak_boost
    # Vectorisation du lissage du gain
    window_size2 = int(sr * 0.05)
    if window_size2 > 1:
        kernel2 = np.ones(window_size2) / window_size2
        smoothed_gain = np.convolve(gain, kernel2, mode='same')
    else:
        smoothed_gain = gain
    y_boosted = y * smoothed_gain
    if logger:
        logger.info(f"Amplification multi-niveaux terminée en {time.time() - start_time:.2f} secondes.")
    return y_boosted

def enhance_voice_clarity(y, sr, gain=0.3, threshold_db=-20, logger=None):
    """
    Améliore la clarté des voix faibles uniquement, avec un seuil strict
    pour éviter tout effet sur les voix fortes.
    """
    if logger:
        logger.info(f"Amélioration de la clarté vocale ultra-sélective: seuil={threshold_db}dB, gain={gain}")
    start_time = time.time()
    threshold_linear = 10 ** (threshold_db / 20)
    b, a = signal.butter(1, [300/(sr/2), 3400/(sr/2)], btype='bandpass')
    y_vocal = signal.filtfilt(b, a, y)
    window_size = int(sr * 0.02)
    amplitude = np.abs(y)
    # Vectorisation du lissage de l'amplitude
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        smoothed_amplitude = np.convolve(amplitude, kernel, mode='same')
    else:
        smoothed_amplitude = amplitude
    safety_threshold = threshold_linear * 0.7
    weak_mask = smoothed_amplitude < safety_threshold
    blend = np.zeros_like(y)
    blend[weak_mask] = gain
    # Vectorisation du lissage du blend
    window_size2 = int(sr * 0.05)
    if window_size2 > 1:
        kernel2 = np.ones(window_size2) / window_size2
        smoothed_blend = np.convolve(blend, kernel2, mode='same')
    else:
        smoothed_blend = blend
    y_enhanced = y + smoothed_blend * y_vocal
    if logger:
        logger.info(f"Clarté vocale terminée en {time.time() - start_time:.2f} secondes.")
    return y_enhanced

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalise l'audio pour éviter la saturation.
    """
    return librosa.util.normalize(audio_data, norm=np.inf, axis=0)

def process_standard_improved(audio_data: np.ndarray, sr: int, logger=None) -> np.ndarray:
    """
    Traitement standard amélioré avec plusieurs étapes pour optimiser les voix faibles.
    """
    weak_threshold_db = -28
    very_weak_threshold_db = -38
    weak_boost = 3.5
    very_weak_boost = 5.0
    if logger:
        logger.info("Début du traitement audio amélioré")
    start_time = time.time()
    audio_data = spectral_subtraction(audio_data, sr, noise_reduce_factor=1.8, logger=logger)
    logger.info(f"Réduction du bruit terminée en {time.time() - start_time:.2f} secondes.")
    b, a = signal.butter(2, [300/(sr/2), 3400/(sr/2)], btype='bandpass')
    vocal_band = signal.filtfilt(b, a, audio_data)
    logger.info(f"Filtrage passe-bande terminé en {time.time() - start_time:.2f} secondes.")
    enhanced = multi_level_amplification(
        audio_data, sr, 
        weak_threshold_db=weak_threshold_db,
        very_weak_threshold_db=very_weak_threshold_db,
        weak_boost=weak_boost, 
        very_weak_boost=very_weak_boost,
        logger=logger
    )
    enhanced = dynamic_range_compression(
        enhanced, sr, 
        threshold_db=-30, 
        ratio=3.0,
        attack_ms=10.0, 
        release_ms=100.0,
        logger=logger
    )
    enhanced = enhance_voice_clarity(
        enhanced, sr, 
        threshold_db=-25, 
        gain=0.4,
        logger=logger
    )
    enhanced = normalize_audio(enhanced)
    logger.info(f"Traitement audio complet terminé en {time.time() - start_time:.2f} secondes.")
    return enhanced

def main():
    # Ajout d'options de ligne de commande
    parser = argparse.ArgumentParser(description="Programme d'amélioration audio optimisé pour voix faibles")
    
    # Options existantes
    parser.add_argument('--input', '-i', required=True, help='Chemin vers le fichier audio à traiter')
    parser.add_argument('--output', '-o', help='Chemin de sortie (optionnel)')
    parser.add_argument('--voice-boost', '-v', type=float, default=3.5,
                       help='Niveau d\'amplification des voix faibles (1.0-10.0)')
    parser.add_argument('--ultra-boost', '-u', type=float, default=5.0,
                       help='Niveau d\'amplification des voix très faibles (1.0-15.0)')
    parser.add_argument('--threshold', '-t', type=float, default=-28,
                       help='Seuil en dB pour les voix faibles (-35 à -15)')
    parser.add_argument('--verbose', '-d', action='store_true', help='Afficher les messages de débogage')
    
    # Nouvelles options
    parser.add_argument('--noise-reduction', '-n', type=float, default=1.8,
                       help='Facteur de réduction du bruit (0.5-3.0)')
    parser.add_argument('--compression', '-c', type=float, default=3.0,
                       help='Ratio de compression (1.0-8.0)')
    parser.add_argument('--transcribe', '-tr', action='store_true', 
                       help='Activer la transcription après le traitement')
    parser.add_argument('--export-json', '-j', action='store_true',
                       help='Exporter les métadonnées au format JSON')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    # Gérer le chemin de sortie
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_enhanced{ext}"
    
    try:
        # Traitement audio amélioré
        logger.info(f"Traitement de {args.input}")
        y, sr = librosa.load(args.input, sr=None)
        
        # Appliquer le traitement standard amélioré
        y_enhanced = process_standard_improved(
            y, sr, 
            logger=logger
        )
        
        # Sauvegarder le résultat
        sf.write(args.output, y_enhanced, sr)
        logger.info(f"Audio amélioré sauvegardé dans {args.output}")
        
        # Transcription si demandée
        if args.transcribe:
            logger.info("Transcription en cours...")
            transcript_path = f"{os.path.splitext(args.output)[0]}_transcript.txt"
            output_json = f"{os.path.splitext(args.output)[0]}_transcript.json" if args.export_json else None
            
            # Effectuer la transcription
            text, transcription_data = transcribe_audio_improved(args.output, output_json=output_json)
            
            # Sauvegarder la transcription
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Transcription sauvegardée dans {transcript_path}")
            
            if args.export_json:
                logger.info(f"Métadonnées de transcription sauvegardées dans {output_json}")
        
        logger.info("Traitement terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 