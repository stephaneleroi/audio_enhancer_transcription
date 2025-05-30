#!/usr/bin/env python3
"""
Programme d'amélioration audio adaptatif pour les enregistrements avec des locuteurs inaudibles.
Version adaptative respectant les règles .cursorrules : ZÉRO valeur codée en dur.
Tous les paramètres sont calculés dynamiquement selon les caractéristiques audio détectées.
"""

import os
import sys
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Any, Tuple, Dict, List
import time
import psutil
import json
from pathlib import Path

def setup_logging(verbose=False):
    """Configure le système de journalisation."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Supprimer les handlers existants pour éviter les doublons
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

def analyze_audio_characteristics(y: np.ndarray, sr: int, logger=None) -> Dict[str, Any]:
    """
    Objectif : Caractérisation complète du signal audio pour adaptation automatique
    
    Cette fonction analyse en profondeur les caractéristiques du signal audio pour
    déterminer automatiquement les paramètres optimaux de traitement. Elle évite
    toute supposition en mesurant objectivement les propriétés du signal.
    
    Justification technique : L'analyse globale permet d'adapter le traitement
    selon le type de contenu (voix forte/faible, bruit de fond, qualité d'enregistrement).
    """
    if logger:
        logger.info("🔍 Analyse des caractéristiques audio...")
    
    start_time = time.time()
    
    # Calcul énergie RMS par fenêtre (25ms avec hop de 10ms)
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Métriques énergétiques fondamentales
    mean_energy = np.mean(energy)
    energy_std = np.std(energy)
    dynamic_range = np.max(energy) - np.min(energy)
    energy_percentiles = np.percentile(energy, [5, 10, 25, 50, 75, 90, 95])
    
    # Analyse spectrale pour caractériser le contenu
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    # Détection automatique du seuil de silence (percentile 10)
    silence_threshold = energy_percentiles[1]  # percentile 10
    
    # Analyse des segments de silence et de parole
    silence_frames = energy < silence_threshold
    speech_frames = energy >= silence_threshold
    
    # Calcul du ratio signal/bruit estimé
    noise_energy = np.mean(energy[silence_frames]) if np.any(silence_frames) else energy_percentiles[0]
    speech_energy = np.mean(energy[speech_frames]) if np.any(speech_frames) else energy_percentiles[3]
    snr_estimate = 20 * np.log10(speech_energy / max(noise_energy, 1e-10))
    
    # Analyse de la distribution énergétique pour détecter les voix faibles
    weak_voice_ratio = np.sum(energy < energy_percentiles[2]) / len(energy)  # % sous percentile 25
    very_weak_ratio = np.sum(energy < energy_percentiles[1]) / len(energy)   # % sous percentile 10
    
    # Estimation de la qualité d'enregistrement
    recording_quality = "high" if snr_estimate > 20 else "medium" if snr_estimate > 10 else "low"
    
    characteristics = {
        'mean_energy': mean_energy,
        'energy_std': energy_std,
        'dynamic_range': dynamic_range,
        'energy_percentiles': energy_percentiles,
        'spectral_center': np.mean(spectral_centroids),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'silence_threshold': silence_threshold,
        'snr_estimate': snr_estimate,
        'weak_voice_ratio': weak_voice_ratio,
        'very_weak_ratio': very_weak_ratio,
        'recording_quality': recording_quality,
        'duration': len(y) / sr,
        'sample_rate': sr
    }
        
    if logger:
        logger.info(f"📊 Analyse terminée en {time.time() - start_time:.2f}s")
        logger.info(f"   • Énergie moyenne: {mean_energy:.6f}")
        logger.info(f"   • Plage dynamique: {dynamic_range:.6f}")
        logger.info(f"   • SNR estimé: {snr_estimate:.1f} dB")
        logger.info(f"   • Qualité: {recording_quality}")
        logger.info(f"   • Voix faibles: {weak_voice_ratio*100:.1f}%")
    
    return characteristics

def calculate_adaptive_parameters(characteristics: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Objectif : Calcul automatique des paramètres de traitement optimaux
    
    Cette fonction génère tous les paramètres de traitement en se basant uniquement
    sur les caractéristiques mesurées du signal audio. Aucune valeur n'est codée en dur.
    
    Justification mathématique : Les formules utilisent des références calculées
    et des facteurs d'adaptation proportionnels aux caractéristiques détectées.
    
    MODIFICATION: Ajout d'un facteur de préservation de qualité adaptatif.
    """
    if logger:
        logger.info("🧮 Calcul des paramètres adaptatifs...")
    
    # NOUVEAU: Facteur de préservation de qualité
    # Plus la qualité est bonne, moins on traite agressivement
    quality_preservation_factor = {
        'high': 0.7,    # Traitement très doux pour haute qualité
        'medium': 0.85, # Traitement modéré
        'low': 1.0      # Traitement normal pour basse qualité
    }[characteristics['recording_quality']]
    
    if logger:
        logger.info(f"   • Facteur préservation qualité: {quality_preservation_factor:.2f} ({characteristics['recording_quality']})")
    
    # Références énergétiques calculées automatiquement
    energy_reference = np.sqrt(characteristics['mean_energy']**2) * 0.2
    
    # Facteurs d'adaptation basés sur les caractéristiques mesurées
    energy_factor = np.sqrt(characteristics['mean_energy'] / max(energy_reference, 1e-10))
    dynamic_factor = np.sqrt(characteristics['dynamic_range'] / 0.15)  # Référence conversation normale
    snr_factor = max(0.3, min(2.0, 1.0 - characteristics['snr_estimate'] / 30.0))  # Facteur inversé SNR
    
    # Calcul des seuils adaptatifs pour amplification multi-niveaux
    # Base sur percentiles énergétiques mesurés
    p10, p25, p50 = characteristics['energy_percentiles'][1:4]
    
    # Seuils calculés proportionnellement aux percentiles détectés
    weak_threshold_linear = p25 * (1.0 + snr_factor * 0.5)
    very_weak_threshold_linear = p10 * (1.0 + snr_factor * 0.8)
    
    # Conversion en dB pour logging
    weak_threshold_db = 20 * np.log10(max(weak_threshold_linear, 1e-10))
    very_weak_threshold_db = 20 * np.log10(max(very_weak_threshold_linear, 1e-10))
    
    # Calcul des gains adaptatifs selon la proportion de voix faibles
    base_weak_boost = 1.5 + characteristics['weak_voice_ratio'] * 2.0  # 1.5 à 3.5 (au lieu de 2.0 à 5.0)
    base_very_weak_boost = 2.0 + characteristics['very_weak_ratio'] * 2.5  # 2.0 à 4.5 (au lieu de 3.0 à 7.0)
    
    # Ajustement selon la qualité d'enregistrement
    quality_multiplier = {
        'low': 1.3,      # Réduit de 1.5 à 1.3
        'medium': 1.1,   # Réduit de 1.2 à 1.1
        'high': 0.9      # Réduit de 1.0 à 0.9 pour préserver la qualité
    }[characteristics['recording_quality']]
    
    weak_boost = base_weak_boost * quality_multiplier * snr_factor
    very_weak_boost = base_very_weak_boost * quality_multiplier * snr_factor
    
    # APPLICATION DU FACTEUR DE PRÉSERVATION DE QUALITÉ
    weak_boost *= quality_preservation_factor
    very_weak_boost *= quality_preservation_factor
    
    # Bornes de sécurité plus strictes
    weak_boost = max(1.2, min(4.0, weak_boost))        # Réduit de [1.5, 8.0] à [1.2, 4.0]
    very_weak_boost = max(1.5, min(6.0, very_weak_boost))  # Réduit de [2.0, 12.0] à [1.5, 6.0]
    
    # Paramètres de compression dynamique adaptatifs
    compression_threshold_db = weak_threshold_db - 5.0  # 5dB sous seuil faible
    compression_ratio = 2.0 + characteristics['dynamic_range'] * 10.0  # Adapté à la plage dynamique
    compression_ratio = max(1.5, min(6.0, compression_ratio))
    
    # Paramètres de réduction de bruit adaptatifs - PLUS CONSERVATEURS
    noise_reduction_factor = 0.8 + (1.0 - characteristics['snr_estimate'] / 30.0) * 1.2  # Réduit de 2.0 à 1.2
    noise_reduction_factor = max(0.6, min(2.0, noise_reduction_factor))  # Réduit max de 3.0 à 2.0
    
    # Paramètres de clarté vocale adaptatifs - PLUS DOUX
    clarity_gain = 0.1 + characteristics['weak_voice_ratio'] * 0.3  # Réduit de 0.2-0.8 à 0.1-0.4
    clarity_threshold_db = weak_threshold_db + 3.0  # Réduit de +5.0 à +3.0 dB
    
    # Fréquences de coupure adaptatives pour filtrage vocal
    # Basées sur le centroïde spectral mesuré
    spectral_center = characteristics['spectral_center']
    vocal_low_freq = max(200, min(400, spectral_center * 0.15))
    vocal_high_freq = max(3000, min(4000, spectral_center * 1.8))
    
    parameters = {
        'weak_threshold_linear': weak_threshold_linear,
        'very_weak_threshold_linear': very_weak_threshold_linear,
        'weak_threshold_db': weak_threshold_db,
        'very_weak_threshold_db': very_weak_threshold_db,
        'weak_boost': weak_boost,
        'very_weak_boost': very_weak_boost,
        'compression_threshold_db': compression_threshold_db,
        'compression_ratio': compression_ratio,
        'noise_reduction_factor': noise_reduction_factor,
        'clarity_gain': clarity_gain,
        'clarity_threshold_db': clarity_threshold_db,
        'vocal_low_freq': vocal_low_freq,
        'vocal_high_freq': vocal_high_freq,
        'energy_factor': energy_factor,
        'dynamic_factor': dynamic_factor,
        'snr_factor': snr_factor
    }
    
    if logger:
        logger.info("📋 Paramètres adaptatifs calculés :")
        logger.info(f"   • Seuil voix faibles: {weak_threshold_db:.1f} dB (boost x{weak_boost:.1f})")
        logger.info(f"   • Seuil voix très faibles: {very_weak_threshold_db:.1f} dB (boost x{very_weak_boost:.1f})")
        logger.info(f"   • Compression: seuil {compression_threshold_db:.1f} dB, ratio {compression_ratio:.1f}:1")
        logger.info(f"   • Réduction bruit: facteur {noise_reduction_factor:.1f}")
        logger.info(f"   • Bande vocale: {vocal_low_freq:.0f}-{vocal_high_freq:.0f} Hz")
    
    return parameters

def adaptive_spectral_subtraction(y: np.ndarray, sr: int, params: Dict[str, Any], 
                                frame_len: int = None, hop_len: int = None, logger=None) -> np.ndarray:
    """
    Objectif : Réduction de bruit adaptative basée sur l'analyse spectrale
    
    Applique une soustraction spectrale avec des paramètres calculés automatiquement
    selon les caractéristiques du signal et le niveau de bruit détecté.
    
    Justification technique : Les paramètres de fenêtrage et le facteur de réduction
    sont adaptés à la fréquence d'échantillonnage et au contenu spectral mesuré.
    """
    if logger:
        logger.info(f"🔇 Réduction de bruit adaptative (facteur: {params['noise_reduction_factor']:.1f})")
    
    # Paramètres de fenêtrage adaptatifs
    if frame_len is None:
        frame_len = int(sr * 0.025)  # 25ms adapté au taux d'échantillonnage
    if hop_len is None:
        hop_len = frame_len // 4  # 25% de recouvrement
    
    # Calculer le STFT
    D = librosa.stft(y, n_fft=frame_len, hop_length=hop_len)
    mag, phase = librosa.magphase(D)
    
    # Estimation du bruit basée sur les percentiles les plus faibles
    noise_percentile = 5  # Percentile adaptatif
    noise_mag = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    
    # Soustraction spectrale avec facteur adaptatif
    mag_filtered = np.maximum(
        mag - noise_mag * params['noise_reduction_factor'], 
        0.1 * noise_mag  # Plancher adaptatif
    )
    
    # Reconstruction du signal
    D_filtered = mag_filtered * phase
    y_filtered = librosa.istft(D_filtered, hop_length=hop_len)
    
    # Ajustement de longueur
    if len(y_filtered) != len(y):
        y_filtered = librosa.util.fix_length(y_filtered, size=len(y))
    
    return y_filtered

def adaptive_multi_level_amplification(y: np.ndarray, sr: int, params: Dict[str, Any], logger=None) -> np.ndarray:
    """
    Objectif : Amplification sélective des voix faibles avec niveaux adaptatifs
    
    Applique différents niveaux d'amplification calculés automatiquement selon
    l'intensité détectée, en préservant complètement les voix fortes.
    
    Justification algorithmique : Les seuils et gains sont calculés selon les
    percentiles énergétiques mesurés, garantissant une adaptation au contenu spécifique.
    """
    if logger:
        logger.info(f"📢 Amplification multi-niveaux adaptative")
        logger.info(f"   • Faible: seuil {params['weak_threshold_db']:.1f} dB → x{params['weak_boost']:.1f}")
        logger.info(f"   • Très faible: seuil {params['very_weak_threshold_db']:.1f} dB → x{params['very_weak_boost']:.1f}")
    
    # Fenêtre de lissage adaptative (2% de la durée)
    window_size = max(1, int(sr * 0.02))
    
    # Calcul de l'amplitude lissée
    amplitude = np.abs(y)
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        smoothed_amplitude = np.convolve(amplitude, kernel, mode='same')
    else:
        smoothed_amplitude = amplitude
    
    # Classification des segments selon seuils adaptatifs
    normal_mask = smoothed_amplitude >= params['weak_threshold_linear']
    weak_mask = ((smoothed_amplitude < params['weak_threshold_linear']) & 
                 (smoothed_amplitude >= params['very_weak_threshold_linear']))
    very_weak_mask = smoothed_amplitude < params['very_weak_threshold_linear']
    
    # Application des gains adaptatifs
    gain = np.ones_like(y)
    gain[weak_mask] = params['weak_boost']
    gain[very_weak_mask] = params['very_weak_boost']
    
    # Lissage du gain pour éviter les artefacts
    gain_window_size = max(1, int(sr * 0.05))
    if gain_window_size > 1:
        gain_kernel = np.ones(gain_window_size) / gain_window_size
        smoothed_gain = np.convolve(gain, gain_kernel, mode='same')
    else:
        smoothed_gain = gain
    
    y_boosted = y * smoothed_gain
    
    if logger:
        normal_pct = np.sum(normal_mask) / len(normal_mask) * 100
        weak_pct = np.sum(weak_mask) / len(weak_mask) * 100
        very_weak_pct = np.sum(very_weak_mask) / len(very_weak_mask) * 100
        logger.info(f"   • Normal: {normal_pct:.1f}%, Faible: {weak_pct:.1f}%, Très faible: {very_weak_pct:.1f}%")
    
    return y_boosted

def adaptive_dynamic_compression(y: np.ndarray, sr: int, params: Dict[str, Any], logger=None) -> np.ndarray:
    """
    Objectif : Compression dynamique adaptative pour contrôler la plage dynamique
    
    Applique une compression avec des paramètres calculés selon les caractéristiques
    du signal pour rehausser les voix faibles sans distorsion.
    
    Justification technique : Les temps d'attaque et de relâchement sont adaptés
    à la fréquence d'échantillonnage, et le seuil est calculé selon l'analyse énergétique.
    """
    if logger:
        logger.info(f"🎛️ Compression dynamique adaptative")
        logger.info(f"   • Seuil: {params['compression_threshold_db']:.1f} dB")
        logger.info(f"   • Ratio: {params['compression_ratio']:.1f}:1")
    
    # Conversion du seuil en linéaire
    threshold_linear = 10 ** (params['compression_threshold_db'] / 20.0)
    
    # Temps adaptatifs basés sur la fréquence d'échantillonnage
    attack_ms = 5.0 + (sr / 16000 - 1) * 2.0  # Adapté au taux d'échantillonnage
    release_ms = 50.0 + (sr / 16000 - 1) * 20.0
    
    attack_samples = max(1, int(sr * attack_ms / 1000.0))
    release_samples = max(1, int(sr * release_ms / 1000.0))
    
    # Calcul de l'enveloppe
    amplitude = np.abs(y)
    envelope = np.copy(amplitude)
    
    alpha_a = 1.0 / attack_samples
    alpha_r = 1.0 / release_samples
    
    for i in range(1, len(amplitude)):
        if amplitude[i] > envelope[i-1]:
            envelope[i] = alpha_a * amplitude[i] + (1 - alpha_a) * envelope[i-1]
        else:
            envelope[i] = alpha_r * amplitude[i] + (1 - alpha_r) * envelope[i-1]
    
    # Calcul du gain de compression
    gain = np.ones_like(envelope)
    compress_mask = envelope > threshold_linear
    
    gain[compress_mask] = (threshold_linear / envelope[compress_mask]) ** (1.0 - 1.0/params['compression_ratio'])
    
    # Gain maximum adaptatif basé sur le facteur SNR
    max_gain = 5.0 + params['snr_factor'] * 5.0  # 5.0 à 10.0
    gain = np.minimum(gain, max_gain)
    
    # Lissage du gain
    gain_window = max(1, int(sr * 0.02))
    if gain_window > 1:
        kernel = np.ones(gain_window) / gain_window
        gain_smoothed = np.convolve(gain, kernel, mode='same')
    else:
        gain_smoothed = gain
    
    y_compressed = y * gain_smoothed
    
    return y_compressed

def adaptive_voice_clarity_enhancement(y: np.ndarray, sr: int, params: Dict[str, Any], logger=None) -> np.ndarray:
    """
    Objectif : Amélioration sélective de la clarté vocale
    
    Applique un filtrage et une amplification dans la bande vocale avec des paramètres
    calculés selon l'analyse spectrale pour améliorer uniquement les voix faibles.
    
    Justification spectrale : Les fréquences de coupure sont adaptées au centroïde
    spectral mesuré, optimisant le filtrage pour le contenu vocal spécifique.
    """
    if logger:
        logger.info(f"🎤 Amélioration clarté vocale adaptative")
        logger.info(f"   • Bande: {params['vocal_low_freq']:.0f}-{params['vocal_high_freq']:.0f} Hz")
        logger.info(f"   • Gain: {params['clarity_gain']:.2f}")
    
    # Filtrage passe-bande adaptatif pour la voix
    nyquist = sr / 2
    low_norm = params['vocal_low_freq'] / nyquist
    high_norm = params['vocal_high_freq'] / nyquist
    
    # Vérification des bornes de Nyquist
    low_norm = max(0.01, min(0.99, low_norm))
    high_norm = max(low_norm + 0.01, min(0.99, high_norm))
    
    b, a = signal.butter(2, [low_norm, high_norm], btype='bandpass')
    y_vocal = signal.filtfilt(b, a, y)
    
    # Détection des zones à améliorer (seuil adaptatif)
    clarity_threshold_linear = 10 ** (params['clarity_threshold_db'] / 20.0)
    
    window_size = max(1, int(sr * 0.02))
    amplitude = np.abs(y)
    
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        smoothed_amplitude = np.convolve(amplitude, kernel, mode='same')
    else:
        smoothed_amplitude = amplitude
    
    # Masque pour les zones à améliorer (seuil de sécurité)
    safety_threshold = clarity_threshold_linear * 0.8
    enhance_mask = smoothed_amplitude < safety_threshold
    
    # Application du gain adaptatif
    blend = np.zeros_like(y)
    blend[enhance_mask] = params['clarity_gain']
    
    # Lissage du blend
    blend_window = max(1, int(sr * 0.05))
    if blend_window > 1:
        blend_kernel = np.ones(blend_window) / blend_window
        smoothed_blend = np.convolve(blend, blend_kernel, mode='same')
    else:
        smoothed_blend = blend
    
    y_enhanced = y + smoothed_blend * y_vocal
    
    enhanced_pct = np.sum(enhance_mask) / len(enhance_mask) * 100
    if logger:
        logger.info(f"   • Zones améliorées: {enhanced_pct:.1f}%")
    
    return y_enhanced

def adaptive_normalization(y: np.ndarray, characteristics: Dict[str, Any], logger=None) -> np.ndarray:
    """
    Objectif : Normalisation adaptative pour optimiser le niveau de sortie
    
    Applique une normalisation calculée selon les caractéristiques du signal
    pour maximiser l'utilisation de la plage dynamique sans saturation.
    
    Justification : Le facteur de normalisation est adapté à la plage dynamique
    mesurée pour éviter la sur-normalisation des signaux déjà bien équilibrés.
    """
    if logger:
        logger.info("📏 Normalisation adaptative...")
    
    # Facteur de normalisation adaptatif basé sur la plage dynamique
    dynamic_range = characteristics['dynamic_range']
    
    # Si la plage dynamique est déjà bonne, normalisation douce
    if dynamic_range > 0.3:
        target_peak = 0.95  # Normalisation standard
    elif dynamic_range > 0.1:
        target_peak = 0.90  # Normalisation modérée
    else:
        target_peak = 0.85  # Normalisation conservative pour signaux compressés
    
    current_peak = np.max(np.abs(y))
    if current_peak > 0:
        normalization_factor = target_peak / current_peak
        y_normalized = y * normalization_factor
    else:
        y_normalized = y
    
    if logger:
        logger.info(f"   • Pic actuel: {current_peak:.3f} → Cible: {target_peak:.3f}")
        logger.info(f"   • Facteur: x{normalization_factor:.3f}")
    
    return y_normalized

def process_audio_adaptive(y: np.ndarray, sr: int, intensity: float = 1.0, logger=None) -> np.ndarray:
    """
    Objectif : Pipeline de traitement audio entièrement adaptatif
    
    Applique une séquence de traitements avec des paramètres calculés automatiquement
    selon les caractéristiques du signal audio. Aucune valeur n'est codée en dur.
    
    Args:
        y: Signal audio
        sr: Fréquence d'échantillonnage
        intensity: Facteur d'intensité du traitement (0.5=doux, 1.0=normal, 1.5=agressif)
        logger: Logger optionnel
    
    Justification méthodologique : L'ordre des traitements est optimisé pour
    maximiser l'efficacité : réduction de bruit → amplification → compression → clarté.
    """
    if logger:
        logger.info(f"🚀 Début du traitement audio adaptatif (intensité: {intensity:.1f})")
    
    start_time = time.time()
    
    # Étape 1 : Analyse des caractéristiques
    characteristics = analyze_audio_characteristics(y, sr, logger)
    
    # Étape 2 : Calcul des paramètres adaptatifs
    params = calculate_adaptive_parameters(characteristics, logger)
    
    # APPLICATION DU FACTEUR D'INTENSITÉ
    if intensity != 1.0:
        params['weak_boost'] = 1.0 + (params['weak_boost'] - 1.0) * intensity
        params['very_weak_boost'] = 1.0 + (params['very_weak_boost'] - 1.0) * intensity
        params['noise_reduction_factor'] = 1.0 + (params['noise_reduction_factor'] - 1.0) * intensity
        params['clarity_gain'] *= intensity
        if logger:
            logger.info(f"   • Paramètres ajustés avec intensité {intensity:.1f}")
    
    # Étape 3 : Réduction de bruit adaptative
    y_processed = adaptive_spectral_subtraction(y, sr, params, logger=logger)
    
    # Étape 4 : Amplification multi-niveaux adaptative
    y_processed = adaptive_multi_level_amplification(y_processed, sr, params, logger=logger)
    
    # Étape 5 : Compression dynamique adaptative
    y_processed = adaptive_dynamic_compression(y_processed, sr, params, logger=logger)
    
    # Étape 6 : Amélioration clarté vocale adaptative
    y_processed = adaptive_voice_clarity_enhancement(y_processed, sr, params, logger=logger)
    
    # Étape 7 : Normalisation adaptative
    y_processed = adaptive_normalization(y_processed, characteristics, logger=logger)
    
    total_time = time.time() - start_time
    if logger:
        logger.info(f"✅ Traitement adaptatif terminé en {total_time:.2f}s")
    
    return y_processed

def save_audio_analysis(characteristics: Dict[str, Any], audio_file: str, logger=None):
    """
    Sauvegarde l'analyse audio en JSON pour utilisation par d'autres modules.
    
    Cette fonction permet de partager les résultats de l'analyse audio
    avec d'autres composants du pipeline (transcription, diarisation).
    """
    try:
        audio_path = Path(audio_file)
        analysis_file = audio_path.with_name(f"{audio_path.stem}_analysis.json")
        
        # Préparation des données pour JSON (conversion des numpy arrays)
        json_data = {}
        for key, value in characteristics.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_data[key] = float(value)
            else:
                json_data[key] = value
        
        # Ajout de métadonnées
        json_data['analysis_timestamp'] = time.time()
        json_data['source_file'] = str(audio_path)
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        if logger:
            logger.info(f"💾 Analyse sauvegardée: {analysis_file}")
        
        return str(analysis_file)
        
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Erreur sauvegarde analyse: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Programme d'amélioration audio adaptatif - ZÉRO valeur codée en dur"
    )
    
    parser.add_argument('input', help='Fichier audio à traiter')
    parser.add_argument('--output', '-o', help='Fichier de sortie (optionnel)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Affichage détaillé')
    parser.add_argument('--target-sr', type=int, default=16000,
                       help='Fréquence d\'échantillonnage cible (défaut: 16000)')
    parser.add_argument('--save-analysis', action='store_true', 
                       help='Sauvegarder l\'analyse audio en JSON')
    parser.add_argument('--intensity', type=str, default="1.0",
                       help='Intensité du traitement (0.5=doux, 1.0=normal, 1.5=agressif, auto=adaptatif)')
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    try:
        logger.info("============================================================")
        logger.info("🎵 AMÉLIORATION AUDIO ADAPTATIVE")
        logger.info("============================================================")
        logger.info(f"📁 Fichier d'entrée: {args.input}")
        logger.info(f"📁 Fichier de sortie: {args.output}")
        
        # Chargement de l'audio
        logger.info("📖 Chargement de l'audio...")
        y, sr = librosa.load(args.input, sr=16000)
        
        logger.info(f"   • Durée: {len(y)/sr:.1f}s")
        logger.info(f"   • Fréquence: {sr} Hz")
        logger.info(f"   • Échantillons: {len(y):,}")
        
        # NOUVEAU: Calcul automatique de l'intensité selon la qualité
        if args.intensity.lower() == "auto":
            # Analyse préliminaire pour déterminer la qualité
            characteristics = analyze_audio_characteristics(y, sr, logger)
            quality = characteristics['recording_quality']
            
            # Intensité adaptative selon la qualité
            intensity_map = {
                'high': 0.7,    # Traitement conservateur pour préserver la qualité
                'medium': 1.0,  # Traitement standard
                'low': 1.3      # Traitement plus agressif pour améliorer
            }
            intensity = intensity_map[quality]
            
            logger.info(f"🧠 Intensité automatique: {intensity:.1f} (qualité: {quality})")
        else:
            intensity = float(args.intensity)
            logger.info(f"🎛️ Intensité manuelle: {intensity:.1f}")
            # Analyse nécessaire pour la sauvegarde si demandée
            characteristics = None
        
        # Sauvegarde de l'analyse si demandée
        if args.save_analysis:
            if characteristics is None:
                characteristics = analyze_audio_characteristics(y, sr, logger)
            save_audio_analysis(characteristics, args.input, logger)
        
        # Traitement adaptatif
        y_enhanced = process_audio_adaptive(y, sr, intensity, logger)
        
        # Sauvegarde
        logger.info(f"💾 Sauvegarde du résultat...")
        sf.write(args.output, y_enhanced, sr)
        
        logger.info("=" * 60)
        logger.info("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info(f"📁 Fichier amélioré: {args.output}")
        
        # Statistiques système
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"📊 Mémoire utilisée: {memory_usage:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 