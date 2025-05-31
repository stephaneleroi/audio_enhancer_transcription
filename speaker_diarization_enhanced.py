#!/usr/bin/env python3
"""
Diarisation intelligente inspirée du projet Verbatim
Corrige la sur-segmentation avec post-traitement adaptatif
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Imports pyannote
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    import torch
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️ pyannote.audio non disponible. Installation requise : pip install pyannote.audio")

def setup_logging():
    """Configure le système de journalisation."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def load_audio_analysis(audio_file, logger=None):
    """
    Charge l'analyse audio sauvegardée si disponible.
    """
    audio_path = Path(audio_file)
    
    # Candidats pour le fichier d'analyse
    analysis_candidates = [
        audio_path.with_name(f"{audio_path.stem}_analysis.json"),
        audio_path.with_name(f"{audio_path.stem}.json"),
    ]
    
    # Gestion des fichiers enhanced
    if "_enhanced" in audio_path.stem:
        original_stem = audio_path.stem.replace("_enhanced", "").replace("_adaptive", "")
        analysis_candidates.extend([
            audio_path.with_name(f"{original_stem}_analysis.json"),
            audio_path.with_name(f"{original_stem}.json"),
        ])
    
    for analysis_file in analysis_candidates:
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)
                if logger:
                    logger.info(f"📊 Analyse audio chargée: {analysis_file}")
                return analysis
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Erreur lecture analyse {analysis_file}: {e}")
    
    if logger:
        logger.info("📊 Aucune analyse audio trouvée, utilisation des paramètres par défaut")
    return None

def calculate_verbatim_parameters(analysis=None, logger=None):
    """
    Calcule les paramètres pyannote selon l'approche Verbatim.
    Privilégie la robustesse à la précision pour éviter la sur-segmentation.
    """
    if analysis is None:
        # Paramètres conservateurs pour éviter la sur-segmentation
        min_duration_off = 1.0  # Plus conservateur que défaut
        clustering_threshold = 0.6  # Plus permissif pour fusionner
        min_cluster_size = 8  # Plus petit pour accepter plus de locuteurs
        
        if logger:
            logger.info("🎯 Paramètres Verbatim par défaut (anti sur-segmentation)")
    else:
        # Extraction des caractéristiques
        duration = analysis.get('duration', 600.0)
        snr_estimate = analysis.get('snr_estimate', 15.0)
        dynamic_range = analysis.get('dynamic_range', 0.15)
        
        # Approche Verbatim : privilégier la stabilité
        
        # 1. min_duration_off : Plus long pour éviter les micro-coupures
        if duration > 1800:  # Fichiers longs
            base_min_duration_off = 1.5
        elif duration > 600:  # Fichiers moyens
            base_min_duration_off = 1.0
        else:  # Fichiers courts
            base_min_duration_off = 0.8
        
        # Ajustement selon la qualité (moins bon = plus conservateur)
        if snr_estimate < 15.0:
            quality_factor = 1.3  # Plus conservateur
        elif snr_estimate > 25.0:
            quality_factor = 0.9  # Moins conservateur
        else:
            quality_factor = 1.0
        
        min_duration_off = base_min_duration_off * quality_factor
        
        # 2. clustering_threshold : Plus permissif pour fusionner les voix similaires
        if dynamic_range > 0.2:
            clustering_threshold = 0.55  # Plus strict seulement si très bonne qualité
        else:
            clustering_threshold = 0.65  # Plus permissif par défaut
        
        # 3. min_cluster_size : Adapté à la durée
        if duration > 1800:
            min_cluster_size = 12
        elif duration > 600:
            min_cluster_size = 8
        else:
            min_cluster_size = 5
        
        if logger:
            logger.info(f"🎯 Paramètres Verbatim adaptatifs:")
            logger.info(f"   Durée: {duration/60:.1f}min, SNR: {snr_estimate:.1f}dB")
            logger.info(f"   Facteur qualité: {quality_factor:.2f}")
    
    # Bornes de sécurité Verbatim
    min_duration_off = max(0.5, min(3.0, min_duration_off))
    clustering_threshold = max(0.4, min(0.8, clustering_threshold))
    min_cluster_size = max(3, min(20, min_cluster_size))
    
    params = {
        'min_duration_off': min_duration_off,
        'clustering_threshold': clustering_threshold,
        'min_cluster_size': min_cluster_size,
    }
    
    if logger:
        logger.info(f"   min_duration_off: {min_duration_off:.2f}s (anti micro-coupures)")
        logger.info(f"   clustering_threshold: {clustering_threshold:.3f} (fusion permissive)")
        logger.info(f"   min_cluster_size: {min_cluster_size} (accepte plus de locuteurs)")
    
    return params

def run_verbatim_diarization(audio_file, params, logger=None):
    """
    Exécute la diarisation avec l'approche Verbatim.
    """
    if not PYANNOTE_AVAILABLE:
        raise ImportError("pyannote.audio non disponible")
    
    if logger:
        logger.info("🎭 Initialisation du pipeline Verbatim...")
    
    try:
        # Chargement du pipeline avec configuration Verbatim
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Configuration Verbatim : privilégier la stabilité
        pipeline.instantiate({
            'segmentation': {
                'min_duration_off': params['min_duration_off'],
            },
            'clustering': {
                'method': 'centroid',
                'min_cluster_size': params['min_cluster_size'],
                'threshold': params['clustering_threshold']
            }
        })
        
        if logger:
            logger.info(f"🎯 Configuration Verbatim appliquée:")
            logger.info(f"   Segmentation: min_duration_off={params['min_duration_off']:.2f}s")
            logger.info(f"   Clustering: threshold={params['clustering_threshold']:.3f}, min_size={params['min_cluster_size']}")
        
        # Exécution
        if logger:
            logger.info(f"🎭 Diarisation Verbatim de {audio_file}...")
        
        start_time = time.time()
        diarization = pipeline(audio_file)
        processing_time = time.time() - start_time
        
        if logger:
            logger.info(f"✅ Diarisation brute terminée en {processing_time:.1f}s")
        
        return diarization
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Erreur diarisation Verbatim: {e}")
        raise

def apply_verbatim_postprocessing(diarization, logger=None):
    """
    Post-traitement intelligent inspiré de Verbatim.
    Corrige la sur-segmentation sans détruire la structure.
    """
    if logger:
        logger.info("🧠 Post-traitement intelligent Verbatim...")
    
    # Conversion en liste de segments
    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': segment.start,
            'end': segment.end,
            'duration': segment.duration,
            'speaker': speaker
        })
    
    # Tri par temps de début
    segments.sort(key=lambda x: x['start'])
    
    if logger:
        logger.info(f"📊 Segments bruts: {len(segments)}")
    
    # Étape 1: Fusion des micro-segments (< 0.5s) avec le segment précédent du même locuteur
    processed_segments = []
    for i, segment in enumerate(segments):
        if (segment['duration'] < 0.5 and 
            processed_segments and 
            processed_segments[-1]['speaker'] == segment['speaker'] and
            segment['start'] - processed_segments[-1]['end'] < 1.0):
            # Fusion avec le segment précédent
            processed_segments[-1]['end'] = segment['end']
            processed_segments[-1]['duration'] = processed_segments[-1]['end'] - processed_segments[-1]['start']
        else:
            processed_segments.append(segment.copy())
    
    if logger:
        logger.info(f"🔧 Après fusion micro-segments: {len(processed_segments)}")
    
    # Étape 2: Fusion des segments très proches du même locuteur (< 0.3s d'écart)
    final_segments = []
    for segment in processed_segments:
        if (final_segments and 
            final_segments[-1]['speaker'] == segment['speaker'] and
            segment['start'] - final_segments[-1]['end'] < 0.3):
            # Fusion avec le segment précédent
            final_segments[-1]['end'] = segment['end']
            final_segments[-1]['duration'] = final_segments[-1]['end'] - final_segments[-1]['start']
        else:
            final_segments.append(segment.copy())
    
    if logger:
        logger.info(f"🔧 Après fusion segments proches: {len(final_segments)}")
    
    # Étape 3: Suppression des segments trop courts isolés (< 0.8s)
    validated_segments = []
    for segment in final_segments:
        if segment['duration'] >= 0.8:
            validated_segments.append(segment)
        elif logger:
            logger.debug(f"🗑️ Segment trop court supprimé: {segment['duration']:.1f}s")
    
    if logger:
        logger.info(f"✅ Segments finaux validés: {len(validated_segments)}")
    
    return validated_segments

def format_verbatim_output(segments, logger=None):
    """
    Formate la sortie selon l'approche Verbatim.
    """
    if not segments:
        if logger:
            logger.warning("⚠️ Aucun segment à formater")
        return segments
    
    # Statistiques
    total_duration = sum(seg['duration'] for seg in segments)
    speakers = set(seg['speaker'] for seg in segments)
    
    # Calcul de la répartition par locuteur
    speaker_durations = defaultdict(float)
    speaker_counts = defaultdict(int)
    
    for seg in segments:
        speaker = seg['speaker']
        speaker_durations[speaker] += seg['duration']
        speaker_counts[speaker] += 1
    
    if logger:
        logger.info(f"📊 Résultats Verbatim finaux:")
        logger.info(f"   {len(segments)} segments optimisés")
        logger.info(f"   {len(speakers)} locuteurs: {', '.join(sorted(speakers))}")
        logger.info(f"   Durée totale de parole: {total_duration:.1f}s")
        logger.info(f"   Répartition par locuteur:")
        for speaker in sorted(speakers):
            duration = speaker_durations[speaker]
            count = speaker_counts[speaker]
            percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
            avg_segment = duration / count if count > 0 else 0
            logger.info(f"     {speaker}: {duration:.1f}s ({percentage:.1f}%) - {count} segments (moy: {avg_segment:.1f}s)")
    
    return segments

def save_verbatim_results(segments, audio_file, logger=None):
    """
    Sauvegarde les résultats selon le format Verbatim.
    """
    audio_path = Path(audio_file)
    output_file = audio_path.with_name(f"{audio_path.stem}_enhanced_diarization.txt")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Diarisation Verbatim - Résultats optimisés\n")
            f.write(f"# Fichier: {audio_file}\n")
            f.write(f"# Segments: {len(segments)}\n")
            f.write(f"# Locuteurs: {len(set(seg['speaker'] for seg in segments))}\n")
            f.write("# Approche: Post-traitement intelligent anti sur-segmentation\n")
            f.write("#\n")
            f.write("# Format: [début] -> [fin] (durée) : LOCUTEUR\n")
            f.write("#" + "="*60 + "\n\n")
            
            for seg in segments:
                start_time = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
                end_time = f"{int(seg['end']//60):02d}:{int(seg['end']%60):02d}"
                f.write(f"[{start_time}] -> [{end_time}] ({seg['duration']:.1f}s) : {seg['speaker']}\n")
        
        if logger:
            logger.info(f"💾 Résultats Verbatim sauvegardés: {output_file}")
        
        return output_file
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Erreur sauvegarde Verbatim: {e}")
        return None

def main():
    """Fonction principale Verbatim."""
    parser = argparse.ArgumentParser(
        description="Diarisation intelligente inspirée de Verbatim - Anti sur-segmentation"
    )
    parser.add_argument("audio_file", help="Fichier audio à traiter")
    parser.add_argument("-o", "--output", help="Fichier de sortie (optionnel)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
    
    args = parser.parse_args()
    
    # Configuration du logging
    logger = setup_logging()
    
    if not args.verbose:
        logger.setLevel(logging.WARNING)
    
    # Vérification du fichier d'entrée
    if not os.path.exists(args.audio_file):
        logger.error(f"❌ Fichier non trouvé: {args.audio_file}")
        sys.exit(1)
    
    try:
        logger.info("🎭 DIARISATION VERBATIM - ANTI SUR-SEGMENTATION")
        logger.info("="*60)
        
        # Chargement de l'analyse audio
        analysis = load_audio_analysis(args.audio_file, logger)
        
        # Calcul des paramètres Verbatim
        params = calculate_verbatim_parameters(analysis, logger)
        
        # Exécution de la diarisation Verbatim
        diarization = run_verbatim_diarization(args.audio_file, params, logger)
        
        # Post-traitement intelligent Verbatim
        segments = apply_verbatim_postprocessing(diarization, logger)
        
        # Formatage des résultats
        segments = format_verbatim_output(segments, logger)
        
        # Sauvegarde
        output_file = save_verbatim_results(segments, args.audio_file, logger)
        
        logger.info("="*60)
        logger.info("✅ DIARISATION VERBATIM TERMINÉE AVEC SUCCÈS")
        
        if output_file:
            logger.info(f"📄 Résultats disponibles dans: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale Verbatim: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 