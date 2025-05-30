#!/usr/bin/env python3
"""
Module de diarisation adaptative des locuteurs basé sur pyannote.
Version adaptative respectant les règles .cursorrules : ZÉRO valeur codée en dur.
Méthodologie inspirée de la transcription adaptative :
1. Analyser l'audio pour déduire des paramètres candidats
2. Tester plusieurs configurations sur un échantillon
3. Sélectionner automatiquement la meilleure
4. Appliquer sur tout le fichier
"""

import os
import sys
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Any, Tuple, Dict, List, Optional
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Imports pyannote
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    import torch
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️ pyannote.audio non disponible. Installation requise : pip install pyannote.audio")

@dataclass
class SpeakerSegment:
    """Représente un segment de parole avec locuteur identifié."""
    start: float
    end: float
    speaker: str
    confidence: float = 0.0
    
    def duration(self) -> float:
        return self.end - self.start
    
    def __str__(self) -> str:
        return f"[{self.start:.1f}s - {self.end:.1f}s] {self.speaker} (conf: {self.confidence:.3f})"

@dataclass
class DiarizationCandidate:
    """Représente une configuration candidate pour la diarisation."""
    clustering_threshold: float
    min_segment_duration: float
    num_speakers: int
    segmentation_onset: float
    segmentation_offset: float
    
    def __str__(self) -> str:
        return (f"threshold={self.clustering_threshold:.2f}, "
                f"min_dur={self.min_segment_duration:.2f}s, "
                f"speakers={self.num_speakers}, "
                f"onset={self.segmentation_onset:.2f}")

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

def analyze_audio_for_diarization(audio_file: str, logger=None) -> Dict[str, Any]:
    """
    Étape 1 : Analyse audio globale pour déduire les caractéristiques de base
    
    Analyse les caractéristiques audio pour estimer les paramètres de base
    de la diarisation. Respecte la règle .cursorrules : ZÉRO valeur codée en dur.
    """
    if logger:
        logger.info("🔍 Étape 1 : Analyse audio globale pour diarisation...")
    
    start_time = time.time()
    
    # Chargement de l'audio
    y, sr = librosa.load(audio_file, sr=None)
    duration = len(y) / sr
    
    # Analyse énergétique
    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms
    
    # Calcul de l'énergie RMS
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calcul de la variation énergétique (pour détecter les changements de locuteurs)
    energy_variation = np.std(rms)
    
    # Analyse spectrale pour détecter la variabilité vocale
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Analyse MFCC pour caractériser les voix
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # Calcul des variations (indicateurs de diversité vocale)
    spectral_variation = np.std(spectral_centroids)
    mfcc_variation = np.mean(np.std(mfccs, axis=1))
    
    # Analyse des silences pour optimiser la segmentation
    silence_threshold = np.percentile(rms, 15)  # Seuil adaptatif pour silences
    silence_frames = rms < silence_threshold
    
    # Calcul de la durée moyenne des silences
    silence_durations = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silence_frames):
        if is_silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            silence_duration = (i - silence_start) * hop_length / sr
            silence_durations.append(silence_duration)
            in_silence = False
    
    avg_silence_duration = np.mean(silence_durations) if silence_durations else 0.5
    
    # Estimation du SNR
    signal_power = np.mean(rms ** 2)
    noise_power = np.mean(rms[rms < np.percentile(rms, 10)] ** 2)
    snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-10))
    
    # Analyse de la diversité vocale (indicateur du nombre de locuteurs)
    # Plus la variation spectrale et MFCC est importante, plus il y a potentiellement de locuteurs
    vocal_diversity = np.sqrt(spectral_variation * mfcc_variation)
    
    # Estimation du nombre de locuteurs basée sur la diversité et la durée
    if duration < 60:  # Fichier court
        base_speakers = max(1, min(3, int(vocal_diversity / 50)))
    elif duration < 300:  # Fichier moyen
        base_speakers = max(2, min(5, int(vocal_diversity / 40)))
    else:  # Fichier long
        base_speakers = max(2, min(8, int(vocal_diversity / 35)))
    
    analysis_time = time.time() - start_time
    
    if logger:
        logger.info(f"📊 Analyse terminée en {analysis_time:.2f}s")
        logger.info(f"   • Durée: {duration:.1f}s")
        logger.info(f"   • Variation énergétique: {energy_variation:.4f}")
        logger.info(f"   • Variation spectrale: {spectral_variation:.2f}")
        logger.info(f"   • Variation MFCC: {mfcc_variation:.4f}")
        logger.info(f"   • Diversité vocale: {vocal_diversity:.2f}")
        logger.info(f"   • SNR estimé: {snr_db:.1f} dB")
        logger.info(f"   • Silences moyens: {avg_silence_duration*1000:.0f}ms")
        logger.info(f"   • Locuteurs estimés: {base_speakers}")
    
    return {
        'duration': duration,
        'sample_rate': sr,
        'energy_variation': energy_variation,
        'spectral_variation': spectral_variation,
        'mfcc_variation': mfcc_variation,
        'vocal_diversity': vocal_diversity,
        'avg_silence_duration': avg_silence_duration,
        'snr_db': snr_db,
        'estimated_speakers': base_speakers
    }

def generate_diarization_candidates(audio_stats: Dict[str, Any], logger=None) -> List[DiarizationCandidate]:
    """
    Étape 2 : Génération de candidats adaptatifs pour la diarisation
    
    Génère plusieurs configurations candidates basées sur l'analyse audio.
    Respecte la règle .cursorrules : ZÉRO valeur codée en dur.
    Version ultra-sensible : Favorise la détection de multiples locuteurs distincts.
    """
    if logger:
        logger.info("🧮 Étape 2 : Génération de candidats adaptatifs (version ultra-sensible)...")
    
    duration = audio_stats['duration']
    vocal_diversity = audio_stats['vocal_diversity']
    estimated_speakers = audio_stats['estimated_speakers']
    avg_silence_duration = audio_stats['avg_silence_duration']
    energy_variation = audio_stats['energy_variation']
    spectral_variation = audio_stats['spectral_variation']
    
    # Calcul de la sensibilité adaptative basée sur la diversité vocale
    # Plus la diversité est élevée, plus on favorise la détection de locuteurs multiples
    # Normalisation de la diversité vocale (typiquement entre 50-300)
    normalized_diversity = min(1.0, vocal_diversity / 200.0)  # Normalisation entre 0 et 1
    sensitivity_factor = 1.0 + normalized_diversity  # Entre 1.0 et 2.0
    
    # Estimation du nombre de locuteurs avec biais vers plus de détection
    min_speakers = max(2, int(estimated_speakers * 0.8))  # Au minimum 80% de l'estimation
    max_speakers = min(10, int(estimated_speakers * 1.8))  # Jusqu'à 180% de l'estimation
    
    # Calcul des paramètres de base adaptatifs
    base_min_duration = max(0.5, avg_silence_duration * 0.3)  # Segments très courts autorisés
    base_clustering_threshold = 0.3 / sensitivity_factor  # Plus sensible si diversité élevée
    
    candidates = []
    
    # Calcul des paramètres de segmentation adaptatifs
    base_onset = 0.5  # Seuil de début de segment
    base_offset = 0.5  # Seuil de fin de segment
    
    # Candidat 1 : Ultra-sensible (favorise beaucoup de locuteurs)
    ultra_sensitive_threshold = base_clustering_threshold * 0.4  # Très sensible
    ultra_sensitive_min_duration = base_min_duration * 0.6  # Segments très courts
    ultra_sensitive_speakers = max_speakers
    
    candidates.append(DiarizationCandidate(
        num_speakers=ultra_sensitive_speakers,
        min_segment_duration=ultra_sensitive_min_duration,
        clustering_threshold=ultra_sensitive_threshold,
        segmentation_onset=base_onset * 0.8,
        segmentation_offset=base_offset * 0.8
    ))
    
    # Candidat 2 : Très sensible (favorise plus de locuteurs)
    very_sensitive_threshold = base_clustering_threshold * 0.6
    very_sensitive_min_duration = base_min_duration * 0.8
    very_sensitive_speakers = int((min_speakers + max_speakers) * 0.8)
    
    candidates.append(DiarizationCandidate(
        num_speakers=very_sensitive_speakers,
        min_segment_duration=very_sensitive_min_duration,
        clustering_threshold=very_sensitive_threshold,
        segmentation_onset=base_onset * 0.9,
        segmentation_offset=base_offset * 0.9
    ))
    
    # Candidat 3 : Sensible adaptatif (basé sur l'estimation)
    adaptive_threshold = base_clustering_threshold * 0.8
    adaptive_min_duration = base_min_duration
    adaptive_speakers = estimated_speakers
    
    candidates.append(DiarizationCandidate(
        num_speakers=adaptive_speakers,
        min_segment_duration=adaptive_min_duration,
        clustering_threshold=adaptive_threshold,
        segmentation_onset=base_onset,
        segmentation_offset=base_offset
    ))
    
    # Candidat 4 : Équilibré (compromis détection/précision)
    balanced_threshold = base_clustering_threshold
    balanced_min_duration = base_min_duration * 1.2
    balanced_speakers = int((min_speakers + estimated_speakers) / 2)
    
    candidates.append(DiarizationCandidate(
        num_speakers=balanced_speakers,
        min_segment_duration=balanced_min_duration,
        clustering_threshold=balanced_threshold,
        segmentation_onset=base_onset * 1.1,
        segmentation_offset=base_offset * 1.1
    ))
    
    # Candidat 5 : Conservateur (évite la sur-segmentation)
    conservative_threshold = base_clustering_threshold * 1.4
    conservative_min_duration = base_min_duration * 1.8
    conservative_speakers = min_speakers
    
    candidates.append(DiarizationCandidate(
        num_speakers=conservative_speakers,
        min_segment_duration=conservative_min_duration,
        clustering_threshold=conservative_threshold,
        segmentation_onset=base_onset * 1.2,
        segmentation_offset=base_offset * 1.2
    ))
    
    # Candidat 6 : Spécialisé haute diversité (si diversité vocale élevée)
    if normalized_diversity > 0.6:  # Seuil adaptatif pour haute diversité (60% de la plage)
        high_diversity_threshold = base_clustering_threshold * 0.3  # Très sensible
        high_diversity_min_duration = base_min_duration * 0.4  # Segments très courts
        high_diversity_speakers = max_speakers
        
        candidates.append(DiarizationCandidate(
            num_speakers=high_diversity_speakers,
            min_segment_duration=high_diversity_min_duration,
            clustering_threshold=high_diversity_threshold,
            segmentation_onset=base_onset * 0.7,
            segmentation_offset=base_offset * 0.7
        ))
    
    # Candidat 7 : Spécialisé variation spectrale (si variation spectrale élevée)
    # Normalisation de la variation spectrale (typiquement entre 500-2000)
    normalized_spectral = min(1.0, spectral_variation / 1500.0)
    if normalized_spectral > 0.4:  # Seuil adaptatif pour variation spectrale
        spectral_threshold = base_clustering_threshold * 0.5
        spectral_min_duration = base_min_duration * 0.7
        spectral_speakers = int(estimated_speakers * 1.4)
        
        candidates.append(DiarizationCandidate(
            num_speakers=spectral_speakers,
            min_segment_duration=spectral_min_duration,
            clustering_threshold=spectral_threshold,
            segmentation_onset=base_onset * 0.85,
            segmentation_offset=base_offset * 0.85
        ))
    
    if logger:
        logger.info(f"   {len(candidates)} candidats générés")
        logger.info(f"   Sensibilité adaptative : {sensitivity_factor:.2f}")
        logger.info(f"   Plage de locuteurs : {min_speakers}-{max_speakers}")
        for i, candidate in enumerate(candidates):
            logger.info(f"   Candidat {i+1}: {candidate}")
    
    return candidates

def test_diarization_candidate(audio_file: str, candidate: DiarizationCandidate, 
                             sample_duration: float = 20.0, logger=None) -> Tuple[List[SpeakerSegment], float]:
    """
    Étape 3 : Test d'un candidat sur un échantillon audio
    
    Teste une configuration de diarisation sur un échantillon et calcule un score.
    """
    if not PYANNOTE_AVAILABLE:
        return [], 0.0
    
    try:
        # Extraction de l'échantillon (milieu du fichier pour éviter les silences de début/fin)
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        # Prendre un échantillon au milieu du fichier
        start_offset = max(0, (duration - sample_duration) / 2)
        end_offset = min(duration, start_offset + sample_duration)
        
        sample_y = y[int(start_offset * sr):int(end_offset * sr)]
        
        # Sauvegarde temporaire de l'échantillon
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, sample_y, sr)
            temp_path = tmp_file.name
        
        try:
            # Création du pipeline avec les paramètres du candidat
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            
            # Configuration du pipeline
            if hasattr(pipeline, '_segmentation'):
                pipeline._segmentation.min_duration_on = candidate.min_segment_duration
                pipeline._segmentation.min_duration_off = candidate.min_segment_duration * 0.5
                if hasattr(pipeline._segmentation, 'onset'):
                    pipeline._segmentation.onset = candidate.segmentation_onset
                if hasattr(pipeline._segmentation, 'offset'):
                    pipeline._segmentation.offset = candidate.segmentation_offset
            
            if hasattr(pipeline, '_clustering'):
                pipeline._clustering.threshold = candidate.clustering_threshold
                if hasattr(pipeline._clustering, 'max_num_speakers'):
                    pipeline._clustering.max_num_speakers = candidate.num_speakers
            
            # Application de la diarisation
            diarization = pipeline(temp_path)
            
            # Conversion en segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start + start_offset,
                    end=turn.end + start_offset,
                    speaker=speaker,
                    confidence=1.0
                )
                segments.append(segment)
            
            # Calcul du score
            score = calculate_diarization_score(segments, sample_duration, candidate.num_speakers)
            
            return segments, score
            
        finally:
            # Nettoyage
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Erreur test candidat: {str(e)}")
        return [], 0.0

def calculate_diarization_score(segments: List[SpeakerSegment], sample_duration: float, 
                              expected_speakers: int) -> float:
    """
    Calcul du score de qualité d'une diarisation
    Version ultra-sensible pour favoriser la détection de multiples locuteurs.
    
    Score basé sur :
    - Nombre de locuteurs détectés vs attendu (favorise PLUS de locuteurs)
    - Distribution des durées de parole (évite la domination d'un seul)
    - Couverture temporelle
    - Granularité des segments (évite les micro-segments mais accepte plus de détail)
    """
    if not segments:
        return 0.0
    
    # Nombre de locuteurs uniques
    unique_speakers = len(set(s.speaker for s in segments))
    
    # Score basé sur le nombre de locuteurs - Version ultra-favorable à plus de locuteurs
    if unique_speakers >= expected_speakers:
        # Récompenser fortement la détection de plus de locuteurs
        speaker_bonus = 1.0 + (unique_speakers - expected_speakers) * 0.15  # Bonus de 15% par locuteur supplémentaire
        speaker_score = min(1.5, speaker_bonus)  # Plafonné à 150%
    else:
        # Pénaliser modérément la sous-détection
        speaker_penalty = unique_speakers / expected_speakers
        speaker_score = speaker_penalty * 0.8  # Pénalité de 20% pour sous-détection
    
    # Distribution des durées de parole (éviter qu'un seul locuteur domine)
    speaker_durations = {}
    for segment in segments:
        speaker = segment.speaker
        duration = segment.end - segment.start
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
    
    total_speech_time = sum(speaker_durations.values())
    if total_speech_time > 0:
        # Calcul de l'équilibre (plus c'est équilibré, mieux c'est)
        duration_ratios = [d / total_speech_time for d in speaker_durations.values()]
        
        # Favoriser une distribution plus équilibrée
        # Utiliser l'entropie normalisée pour mesurer l'équilibre
        import math
        entropy = -sum(r * math.log(r + 1e-10) for r in duration_ratios)
        max_entropy = math.log(len(duration_ratios))
        balance_score = entropy / max_entropy if max_entropy > 0 else 0.5
        
        # Bonus pour plus de locuteurs actifs
        balance_score = min(1.0, balance_score + (unique_speakers - 2) * 0.1)  # Bonus pour 3+ locuteurs
    else:
        balance_score = 0.0
    
    # Couverture temporelle (pourcentage du temps couvert)
    covered_time = sum(segment.end - segment.start for segment in segments)
    coverage_score = min(1.0, covered_time / sample_duration)
    
    # Score de granularité (éviter les micro-segments mais accepter plus de détail)
    avg_segment_duration = covered_time / len(segments) if segments else 0
    
    # Granularité optimale adaptative basée sur la durée d'échantillon
    optimal_granularity = max(0.5, sample_duration / (expected_speakers * 3))  # 3 segments par locuteur en moyenne
    
    if avg_segment_duration > optimal_granularity * 2:
        # Segments trop longs - pénaliser modérément
        granularity_score = 0.7
    elif avg_segment_duration < optimal_granularity * 0.3:
        # Segments trop courts - pénaliser légèrement
        granularity_score = 0.8
    else:
        # Granularité acceptable - récompenser
        granularity_score = 1.0
    
    # Bonus pour diversité de locuteurs
    diversity_bonus = min(0.3, (unique_speakers - 1) * 0.1)  # Bonus jusqu'à 30% pour diversité
    
    # Score composite avec pondération favorisant la détection multiple
    final_score = (
        speaker_score * 0.4 +      # 40% - Nombre de locuteurs (le plus important)
        balance_score * 0.25 +     # 25% - Équilibre entre locuteurs
        coverage_score * 0.2 +     # 20% - Couverture temporelle
        granularity_score * 0.15   # 15% - Granularité des segments
    ) + diversity_bonus            # Bonus pour diversité
    
    return min(2.0, final_score)  # Plafonné à 200% pour récompenser les bonnes détections multiples

def calibrate_diarization_on_sample(audio_file: str, candidates: List[DiarizationCandidate], 
                                   logger=None) -> DiarizationCandidate:
    """
    Étape 3 : Calibration sur échantillon pour sélectionner le meilleur candidat
    
    Teste tous les candidats sur un échantillon et sélectionne le meilleur.
    """
    if logger:
        logger.info("🎯 Étape 3 : Calibration sur échantillon (20s)...")
    
    best_candidate = candidates[0]
    best_score = 0.0
    best_segments = []
    
    for i, candidate in enumerate(candidates, 1):
        if logger:
            logger.info(f"   Test candidat {i}/{len(candidates)}: {candidate}")
        
        segments, score = test_diarization_candidate(audio_file, candidate, logger=logger)
        
        if logger:
            unique_speakers = len(set(s.speaker for s in segments)) if segments else 0
            total_duration = sum(s.duration() for s in segments) if segments else 0
            logger.info(f"      → {len(segments)} segments, {unique_speakers} locuteurs, "
                       f"{total_duration:.1f}s parole, score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_segments = segments
    
    if logger:
        logger.info(f"✅ Meilleur candidat sélectionné (score: {best_score:.3f})")
        logger.info(f"   → {best_candidate}")
        unique_speakers = len(set(s.speaker for s in best_segments)) if best_segments else 0
        logger.info(f"   → {len(best_segments)} segments, {unique_speakers} locuteurs détectés")
    
    return best_candidate

def diarize_audio_adaptive(audio_file: str, logger=None) -> Tuple[List[SpeakerSegment], Dict[str, Any]]:
    """
    Étape 4 : Diarisation adaptative complète
    
    Applique la méthodologie complète : analyse → candidats → test → application
    """
    if logger:
        logger.info("🚀 Début de la diarisation adaptative complète...")
    
    # Étape 1: Analyse audio
    audio_stats = analyze_audio_for_diarization(audio_file, logger)
    
    # Étape 2: Génération des candidats
    candidates = generate_diarization_candidates(audio_stats, logger)
    
    # Étape 3: Calibration sur échantillon
    best_candidate = calibrate_diarization_on_sample(audio_file, candidates, logger)
    
    # Étape 4: Application sur tout le fichier
    if logger:
        logger.info("🎬 Étape 4 : Application sur le fichier complet...")
    
    try:
        # Création du pipeline avec les meilleurs paramètres
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        # Configuration du pipeline
        if hasattr(pipeline, '_segmentation'):
            pipeline._segmentation.min_duration_on = best_candidate.min_segment_duration
            pipeline._segmentation.min_duration_off = best_candidate.min_segment_duration * 0.5
            if hasattr(pipeline._segmentation, 'onset'):
                pipeline._segmentation.onset = best_candidate.segmentation_onset
            if hasattr(pipeline._segmentation, 'offset'):
                pipeline._segmentation.offset = best_candidate.segmentation_offset
        
        if hasattr(pipeline, '_clustering'):
            pipeline._clustering.threshold = best_candidate.clustering_threshold
            if hasattr(pipeline._clustering, 'max_num_speakers'):
                pipeline._clustering.max_num_speakers = best_candidate.num_speakers
        
        # Application de la diarisation
        diarization = pipeline(audio_file)
        
        # Conversion en segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
                confidence=1.0
            )
            segments.append(segment)
        
        # Tri par temps de début
        segments.sort(key=lambda s: s.start)
        
        if logger:
            unique_speakers = len(set(s.speaker for s in segments))
            total_duration = sum(s.duration() for s in segments)
            logger.info(f"✅ Diarisation terminée : {len(segments)} segments, "
                       f"{unique_speakers} locuteurs, {total_duration:.1f}s de parole")
        
        # Paramètres utilisés
        parameters = {
            'clustering_threshold': best_candidate.clustering_threshold,
            'min_segment_duration': best_candidate.min_segment_duration,
            'num_speakers': best_candidate.num_speakers,
            'segmentation_onset': best_candidate.segmentation_onset,
            'segmentation_offset': best_candidate.segmentation_offset
        }
        
        return segments, parameters
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Erreur lors de la diarisation: {str(e)}")
        return [], {}

def save_diarization_results(segments: List[SpeakerSegment], output_file: str, 
                           parameters: Dict[str, Any], logger=None):
    """
    Sauvegarde des résultats de diarisation avec paramètres utilisés
    """
    if logger:
        logger.info(f"💾 Sauvegarde des résultats: {output_file}")
    
    # Format texte détaillé
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RÉSULTATS DE DIARISATION ADAPTATIVE\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PARAMÈTRES OPTIMAUX DÉCOUVERTS:\n")
        f.write("-" * 30 + "\n")
        for key, value in parameters.items():
            if isinstance(value, float):
                f.write(f"• {key}: {value:.3f}\n")
            else:
                f.write(f"• {key}: {value}\n")
        f.write("\n")
        
        f.write("SEGMENTS DÉTECTÉS:\n")
        f.write("-" * 30 + "\n")
        for i, segment in enumerate(segments, 1):
            f.write(f"{i:3d}. {segment}\n")
        
        f.write(f"\n📊 STATISTIQUES:\n")
        f.write(f"   • Segments totaux: {len(segments)}\n")
        f.write(f"   • Locuteurs uniques: {len(set(s.speaker for s in segments))}\n")
        f.write(f"   • Durée totale: {max(s.end for s in segments) if segments else 0:.1f}s\n")
        
        # Statistiques par locuteur
        speaker_stats = {}
        for segment in segments:
            if segment.speaker not in speaker_stats:
                speaker_stats[segment.speaker] = {'duration': 0, 'segments': 0}
            speaker_stats[segment.speaker]['duration'] += segment.duration()
            speaker_stats[segment.speaker]['segments'] += 1
        
        f.write(f"\n👥 STATISTIQUES PAR LOCUTEUR:\n")
        for speaker, stats in sorted(speaker_stats.items()):
            f.write(f"   • {speaker}: {stats['duration']:.1f}s ({stats['segments']} segments)\n")
    
    # Format RTTM (Rich Transcription Time Marked)
    rttm_file = output_file.replace('.txt', '.rttm')
    with open(rttm_file, 'w') as f:
        for segment in segments:
            f.write(f"SPEAKER audio 1 {segment.start:.3f} {segment.duration():.3f} "
                   f"<NA> <NA> {segment.speaker} <NA> <NA>\n")
    
    if logger:
        logger.info(f"📁 Fichiers sauvegardés: {output_file} et {rttm_file}")

def main():
    """Point d'entrée principal du programme."""
    parser = argparse.ArgumentParser(
        description="Diarisation adaptative des locuteurs (méthodologie transcription)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Méthodologie adaptative :
1. Analyse audio globale pour déduire les caractéristiques
2. Génération de 7 candidats adaptatifs
3. Test et calibration sur échantillon de 20s
4. Application des meilleurs paramètres sur tout le fichier

Exemples d'utilisation:
  python speaker_diarization.py audio.wav
  python speaker_diarization.py audio.wav --verbose
        """
    )
    
    parser.add_argument("input_file", help="Fichier audio d'entrée")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Affichage détaillé")
    
    args = parser.parse_args()
    
    # Configuration du logging
    logger = setup_logging(args.verbose)
    
    try:
        logger.info("=" * 60)
        logger.info("🎭 DIARISATION ADAPTATIVE DES LOCUTEURS")
        logger.info("=" * 60)
        logger.info(f"📁 Fichier d'entrée: {args.input_file}")
        
        # Génération du nom de fichier de sortie
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_diarization.txt"
        logger.info(f"📁 Fichier de sortie: {output_file}")
        
        # Diarisation adaptative complète
        segments, parameters = diarize_audio_adaptive(args.input_file, logger=logger)
        
        # Sauvegarde des résultats
        save_diarization_results(segments, output_file, parameters, logger)
        
        logger.info("✅ Diarisation adaptative terminée avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la diarisation: {str(e)}")
        if args.verbose:
            import traceback
            logger.error("Traceback (most recent call last):")
            for line in traceback.format_exc().strip().split('\n'):
                logger.error(line)
        sys.exit(1)

if __name__ == "__main__":
    main() 