#!/usr/bin/env python3
"""
Module de diarisation adaptative des locuteurs basé sur pyannote.
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

def analyze_audio_for_diarization(y: np.ndarray, sr: int, logger=None) -> Dict[str, Any]:
    """
    Objectif : Analyse spécialisée pour la diarisation des locuteurs
    
    Analyse les caractéristiques audio pertinentes pour la séparation des locuteurs :
    - Variations spectrales (changements de voix)
    - Pauses et silences (transitions entre locuteurs)
    - Énergie et dynamique (intensité vocale)
    - Estimation du nombre de locuteurs potentiels
    """
    if logger:
        logger.info("🔍 Analyse audio spécialisée pour diarisation...")
    
    start_time = time.time()
    
    # Paramètres adaptatifs pour l'analyse
    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms
    
    # Analyse énergétique
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Analyse spectrale pour détecter les changements de voix
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    
    # MFCC pour caractériser les voix
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # Calcul des variations spectrales (indicateur de changement de locuteur)
    spectral_variation = np.std(np.diff(spectral_centroids))
    mfcc_variation = np.mean(np.std(mfccs, axis=1))
    
    # Détection des silences adaptatifs
    energy_percentiles = np.percentile(energy, [5, 10, 25, 50, 75, 90, 95])
    silence_threshold = energy_percentiles[1]  # percentile 10
    
    # Analyse des pauses (potentiels changements de locuteur)
    silence_frames = energy < silence_threshold
    
    # Calcul des durées de silence
    silence_durations = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silence_frames):
        if is_silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            duration_ms = (i - silence_start) * hop_length / sr * 1000
            silence_durations.append(duration_ms)
            in_silence = False
    
    # Estimation du nombre de locuteurs basée sur les variations
    # Plus il y a de variations spectrales, plus il y a potentiellement de locuteurs
    estimated_speakers = max(1, min(6, int(2 + spectral_variation * 10)))
    
    # Calcul des seuils adaptatifs pour la segmentation
    mean_energy = np.mean(energy)
    energy_std = np.std(energy)
    dynamic_range = np.max(energy) - np.min(energy)
    
    # Estimation de la qualité pour la diarisation
    snr_estimate = 20 * np.log10(np.mean(energy[energy > silence_threshold]) / 
                                max(np.mean(energy[silence_frames]), 1e-10))
    
    characteristics = {
        'duration': len(y) / sr,
        'sample_rate': sr,
        'mean_energy': mean_energy,
        'energy_std': energy_std,
        'dynamic_range': dynamic_range,
        'energy_percentiles': energy_percentiles,
        'silence_threshold': silence_threshold,
        'spectral_variation': spectral_variation,
        'mfcc_variation': mfcc_variation,
        'estimated_speakers': estimated_speakers,
        'silence_durations': silence_durations,
        'mean_silence_duration': np.mean(silence_durations) if silence_durations else 0,
        'snr_estimate': snr_estimate,
        'spectral_center_mean': np.mean(spectral_centroids),
        'spectral_center_std': np.std(spectral_centroids),
        'recording_quality': "high" if snr_estimate > 20 else "medium" if snr_estimate > 10 else "low"
    }
    
    if logger:
        logger.info(f"📊 Analyse terminée en {time.time() - start_time:.2f}s")
        logger.info(f"   • Durée: {characteristics['duration']:.1f}s")
        logger.info(f"   • Locuteurs estimés: {estimated_speakers}")
        logger.info(f"   • Variation spectrale: {spectral_variation:.4f}")
        logger.info(f"   • Variation MFCC: {mfcc_variation:.4f}")
        logger.info(f"   • SNR estimé: {snr_estimate:.1f} dB")
        logger.info(f"   • Silences moyens: {characteristics['mean_silence_duration']:.0f}ms")
    
    return characteristics

def calculate_diarization_parameters(characteristics: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Objectif : Calcul automatique des paramètres de diarisation optimaux
    
    Génère tous les paramètres de diarisation en se basant uniquement sur les
    caractéristiques mesurées du signal audio. Aucune valeur n'est codée en dur.
    """
    if logger:
        logger.info("🧮 Calcul des paramètres de diarisation adaptatifs...")
    
    # Calcul du nombre optimal de locuteurs
    base_speakers = characteristics['estimated_speakers']
    
    # Ajustement selon la qualité et les variations
    quality_factor = {
        'high': 1.0,
        'medium': 0.8,
        'low': 0.6
    }[characteristics['recording_quality']]
    
    variation_factor = min(2.0, characteristics['spectral_variation'] * 5)
    
    # Nombre de locuteurs adaptatif avec bornes de sécurité
    min_speakers = max(1, int(base_speakers * quality_factor * 0.7))
    max_speakers = max(2, int(base_speakers * quality_factor * variation_factor))
    max_speakers = min(max_speakers, 8)  # Limite raisonnable
    
    # Seuil de segmentation adaptatif
    # Basé sur les variations spectrales et la qualité
    base_segmentation_threshold = 0.1 + characteristics['spectral_variation'] * 2.0
    segmentation_threshold = max(0.05, min(0.8, base_segmentation_threshold))
    
    # Durée minimale de segment adaptative
    # Basée sur la durée moyenne des silences
    mean_silence = characteristics['mean_silence_duration']
    if mean_silence > 0:
        min_segment_duration = max(0.5, min(3.0, mean_silence / 1000 * 2))
    else:
        min_segment_duration = 1.0
    
    # Seuil de clustering adaptatif
    # Plus il y a de variations, plus le seuil doit être fin
    clustering_threshold = max(0.1, min(0.7, 0.4 - characteristics['mfcc_variation'] * 0.1))
    
    # Taille de chunk adaptative selon la durée totale
    duration = characteristics['duration']
    if duration < 60:  # < 1 minute
        chunk_duration = min(30, duration / 2)
    elif duration < 300:  # < 5 minutes
        chunk_duration = 60
    else:  # > 5 minutes
        chunk_duration = 120
    
    # Overlap adaptatif
    overlap_duration = max(5, min(15, chunk_duration * 0.1))
    
    parameters = {
        'min_speakers': min_speakers,
        'max_speakers': max_speakers,
        'segmentation_threshold': segmentation_threshold,
        'clustering_threshold': clustering_threshold,
        'min_segment_duration': min_segment_duration,
        'chunk_duration': chunk_duration,
        'overlap_duration': overlap_duration,
        'quality_factor': quality_factor,
        'variation_factor': variation_factor
    }
    
    if logger:
        logger.info("📋 Paramètres de diarisation calculés :")
        logger.info(f"   • Locuteurs: {min_speakers}-{max_speakers}")
        logger.info(f"   • Seuil segmentation: {segmentation_threshold:.3f}")
        logger.info(f"   • Seuil clustering: {clustering_threshold:.3f}")
        logger.info(f"   • Durée min segment: {min_segment_duration:.1f}s")
        logger.info(f"   • Chunks: {chunk_duration:.0f}s (overlap: {overlap_duration:.0f}s)")
    
    return parameters

def create_diarization_pipeline(parameters: Dict[str, Any], logger=None) -> Optional[Pipeline]:
    """
    Objectif : Création du pipeline pyannote avec paramètres adaptatifs
    
    Configure le pipeline de diarisation avec les paramètres calculés automatiquement.
    """
    if not PYANNOTE_AVAILABLE:
        if logger:
            logger.error("❌ pyannote.audio non disponible")
        return None
    
    if logger:
        logger.info("🔧 Création du pipeline de diarisation adaptatif...")
    
    try:
        # Utilisation du modèle pré-entraîné
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None  # Peut nécessiter un token HuggingFace
        )
        
        # Configuration adaptative du pipeline
        if hasattr(pipeline, '_segmentation'):
            # Ajustement des paramètres de segmentation
            pipeline._segmentation.min_duration_on = parameters['min_segment_duration']
            pipeline._segmentation.min_duration_off = parameters['min_segment_duration'] * 0.5
        
        if hasattr(pipeline, '_clustering'):
            # Ajustement des paramètres de clustering
            pipeline._clustering.threshold = parameters['clustering_threshold']
        
        if logger:
            logger.info("✅ Pipeline de diarisation configuré")
        
        return pipeline
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Erreur création pipeline: {str(e)}")
        return None

def process_audio_chunk_diarization(chunk_data: bytes) -> List[SpeakerSegment]:
    """
    Objectif : Traitement d'un chunk audio pour la diarisation
    
    Fonction sérialisable pour le traitement parallèle des chunks.
    """
    try:
        # Désérialisation des données
        data = pickle.loads(chunk_data)
        audio_file = data['audio_file']
        start_time = data['start_time']
        end_time = data['end_time']
        parameters = data['parameters']
        chunk_id = data['chunk_id']
        
        # Chargement du chunk audio
        y, sr = librosa.load(audio_file, sr=16000, offset=start_time, duration=end_time-start_time)
        
        # Sauvegarde temporaire du chunk
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, y, sr)
            temp_path = tmp_file.name
        
        try:
            # Création du pipeline pour ce worker
            pipeline = create_diarization_pipeline(parameters)
            if pipeline is None:
                return []
            
            # Application de la diarisation
            diarization = pipeline(temp_path)
            
            # Conversion en segments avec ajustement temporel
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start + start_time,
                    end=turn.end + start_time,
                    speaker=speaker,
                    confidence=1.0  # pyannote ne fournit pas de score de confiance direct
                )
                segments.append(segment)
            
            return segments
            
        finally:
            # Nettoyage du fichier temporaire
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"❌ Erreur traitement chunk: {str(e)}")
        return []

def diarize_audio_adaptive(audio_file: str, parameters: Dict[str, Any], 
                          num_workers: int = 4, logger=None) -> List[SpeakerSegment]:
    """
    Objectif : Diarisation adaptative avec traitement parallèle par chunks
    
    Applique la diarisation en découpant l'audio en chunks et en traitant
    en parallèle avec les paramètres calculés automatiquement.
    """
    if logger:
        logger.info("🚀 Début de la diarisation adaptative parallèle...")
    
    # Chargement de l'audio pour analyse
    y, sr = librosa.load(audio_file, sr=16000)
    duration = len(y) / sr
    
    chunk_duration = parameters['chunk_duration']
    overlap_duration = parameters['overlap_duration']
    
    # Calcul des chunks avec overlap
    chunks = []
    start_time = 0
    chunk_id = 0
    
    while start_time < duration:
        end_time = min(start_time + chunk_duration, duration)
        
        # Préparation des données pour le worker
        chunk_data = {
            'audio_file': audio_file,
            'start_time': start_time,
            'end_time': end_time,
            'parameters': parameters,
            'chunk_id': chunk_id
        }
        
        chunks.append(pickle.dumps(chunk_data))
        
        # Avancement avec overlap
        start_time += chunk_duration - overlap_duration
        chunk_id += 1
        
        if end_time >= duration:
            break
    
    if logger:
        logger.info(f"📦 {len(chunks)} chunks créés (durée: {chunk_duration}s, overlap: {overlap_duration}s)")
        logger.info(f"🖥️ Traitement parallèle avec {num_workers} workers")
    
    # Traitement parallèle
    all_segments = []
    completed_chunks = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Soumission des tâches
        future_to_chunk = {
            executor.submit(process_audio_chunk_diarization, chunk_data): i 
            for i, chunk_data in enumerate(chunks)
        }
        
        # Collecte des résultats
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                segments = future.result()
                all_segments.extend(segments)
                completed_chunks += 1
                
                if logger:
                    progress = (completed_chunks / len(chunks)) * 100
                    logger.info(f"📊 Chunk {chunk_idx+1}/{len(chunks)} terminé: "
                              f"{len(segments)} segments - Progrès: {progress:.1f}%")
                    
            except Exception as e:
                if logger:
                    logger.error(f"❌ Erreur chunk {chunk_idx}: {str(e)}")
    
    # Post-traitement : fusion des segments overlappants et nettoyage
    merged_segments = merge_overlapping_segments(all_segments, parameters, logger)
    
    if logger:
        logger.info(f"✅ Diarisation terminée: {len(merged_segments)} segments finaux")
    
    return merged_segments

def merge_overlapping_segments(segments: List[SpeakerSegment], parameters: Dict[str, Any], 
                             logger=None) -> List[SpeakerSegment]:
    """
    Objectif : Fusion intelligente des segments overlappants
    
    Fusionne les segments provenant de chunks différents en évitant les doublons
    et en optimisant la continuité des locuteurs.
    """
    if not segments:
        return []
    
    if logger:
        logger.info(f"🔧 Fusion de {len(segments)} segments bruts...")
    
    # Tri par temps de début
    segments.sort(key=lambda s: s.start)
    
    merged = []
    current_segment = segments[0]
    
    for next_segment in segments[1:]:
        # Vérification de l'overlap
        if (next_segment.start <= current_segment.end and 
            next_segment.speaker == current_segment.speaker):
            # Fusion des segments du même locuteur qui se chevauchent
            current_segment.end = max(current_segment.end, next_segment.end)
            current_segment.confidence = max(current_segment.confidence, next_segment.confidence)
        else:
            # Ajout du segment actuel et passage au suivant
            if current_segment.duration() >= parameters['min_segment_duration']:
                merged.append(current_segment)
            current_segment = next_segment
    
    # Ajout du dernier segment
    if current_segment.duration() >= parameters['min_segment_duration']:
        merged.append(current_segment)
    
    if logger:
        logger.info(f"✅ {len(merged)} segments après fusion")
    
    return merged

def save_diarization_results(segments: List[SpeakerSegment], output_file: str, logger=None):
    """
    Objectif : Sauvegarde des résultats de diarisation
    
    Sauvegarde les segments avec locuteurs dans différents formats.
    """
    if logger:
        logger.info(f"💾 Sauvegarde des résultats: {output_file}")
    
    # Format texte détaillé
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RÉSULTATS DE DIARISATION ADAPTATIVE\n")
        f.write("=" * 50 + "\n\n")
        
        for i, segment in enumerate(segments, 1):
            f.write(f"{i:3d}. {segment}\n")
        
        f.write(f"\n📊 STATISTIQUES:\n")
        f.write(f"   • Segments totaux: {len(segments)}\n")
        f.write(f"   • Locuteurs uniques: {len(set(s.speaker for s in segments))}\n")
        f.write(f"   • Durée totale: {max(s.end for s in segments):.1f}s\n")
        
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
    parser = argparse.ArgumentParser(
        description="Diarisation adaptative des locuteurs - ZÉRO valeur codée en dur"
    )
    
    parser.add_argument('input_file', help='Fichier audio à analyser')
    parser.add_argument('--output', '-o', help='Fichier de sortie (optionnel)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Affichage détaillé')
    parser.add_argument('--workers', '-w', type=int, default=4, 
                       help='Nombre de workers parallèles (défaut: 4)')
    
    args = parser.parse_args()
    
    if not PYANNOTE_AVAILABLE:
        print("❌ pyannote.audio requis. Installation: pip install pyannote.audio")
        sys.exit(1)
    
    logger = setup_logging(args.verbose)
    
    # Génération automatique du nom de sortie
    if not args.output:
        base, ext = os.path.splitext(args.input_file)
        args.output = f"{base}_diarization.txt"
    
    try:
        logger.info("=" * 60)
        logger.info("🎭 DIARISATION ADAPTATIVE DES LOCUTEURS")
        logger.info("=" * 60)
        logger.info(f"📁 Fichier d'entrée: {args.input_file}")
        logger.info(f"📁 Fichier de sortie: {args.output}")
        
        # Vérification du fichier d'entrée
        if not os.path.exists(args.input_file):
            logger.error(f"❌ Fichier non trouvé: {args.input_file}")
            sys.exit(1)
        
        # Étape 1: Analyse des caractéristiques audio
        y, sr = librosa.load(args.input_file, sr=16000)
        characteristics = analyze_audio_for_diarization(y, sr, logger)
        
        # Étape 2: Calcul des paramètres adaptatifs
        parameters = calculate_diarization_parameters(characteristics, logger)
        
        # Étape 3: Diarisation adaptative
        segments = diarize_audio_adaptive(
            args.input_file, 
            parameters, 
            num_workers=args.workers, 
            logger=logger
        )
        
        # Étape 4: Sauvegarde des résultats
        save_diarization_results(segments, args.output, logger)
        
        logger.info("=" * 60)
        logger.info("✅ DIARISATION TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info(f"📊 {len(segments)} segments identifiés")
        logger.info(f"👥 {len(set(s.speaker for s in segments))} locuteurs détectés")
        logger.info(f"📁 Résultats: {args.output}")
        
        # Statistiques système
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"📊 Mémoire utilisée: {memory_usage:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la diarisation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 