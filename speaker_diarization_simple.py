#!/usr/bin/env python3
"""
Diarisation simple et efficace des locuteurs
Approche directe sans calibration inadéquate sur échantillon
Avec optimisation optionnelle basée sur l'analyse audio du pré-traitement
"""

import os
import sys
import argparse
import logging
import time
import json
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

def setup_logging():
    """Configure le système de journalisation."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Supprimer les handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_audio_analysis(audio_file: str, logger=None):
    """
    Charge l'analyse audio du pré-traitement si disponible.
    
    Recherche un fichier JSON d'analyse correspondant au fichier audio.
    Cette approche évite de refaire l'analyse et utilise intelligemment
    les résultats du pré-traitement.
    """
    audio_path = Path(audio_file)
    
    # Recherche de fichiers d'analyse possibles
    analysis_candidates = [
        audio_path.with_suffix('.json'),  # même nom avec .json
        audio_path.with_name(f"{audio_path.stem}_analysis.json"),  # nom_analysis.json
        audio_path.with_name(f"{audio_path.stem}_enhanced_analysis.json"),  # enhanced_analysis.json
    ]
    
    # Si le fichier est un fichier "enhanced", chercher aussi l'analyse du fichier original
    if "_adaptive_enhanced" in audio_path.stem:
        original_stem = audio_path.stem.replace("_adaptive_enhanced", "")
        analysis_candidates.extend([
            audio_path.with_name(f"{original_stem}.json"),
            audio_path.with_name(f"{original_stem}_analysis.json"),
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

def optimize_diarization_parameters(analysis=None, logger=None):
    """
    Optimise les paramètres de diarisation basés sur l'analyse audio.
    
    Contrairement à l'approche "adaptative" défaillante, cette fonction :
    1. Utilise les paramètres par défaut optimisés comme base
    2. Applique des ajustements LÉGERS basés sur l'analyse
    3. Ne change jamais radicalement les paramètres
    4. Garde toujours des bornes de sécurité
    """
    # Paramètres par défaut optimisés (base solide)
    base_params = {
        'clustering_threshold': None,  # Laissé à pyannote
        'min_duration_on': 0.5,       # Durée minimale de parole
        'min_duration_off': 0.1,      # Durée minimale de silence
        'onset': 0.5,                 # Seuil de début de segment
        'offset': 0.5                 # Seuil de fin de segment
    }
    
    if analysis is None:
        if logger:
            logger.info("🎯 Utilisation des paramètres par défaut optimisés")
        return base_params
    
    # Ajustements LÉGERS basés sur l'analyse
    optimized_params = base_params.copy()
    
    try:
        # Ajustement basé sur la qualité d'enregistrement
        recording_quality = analysis.get('recording_quality', 'medium')
        snr_estimate = analysis.get('snr_estimate', 15.0)
        weak_voice_ratio = analysis.get('weak_voice_ratio', 0.3)
        
        if logger:
            logger.info(f"📊 Qualité audio: {recording_quality}, SNR: {snr_estimate:.1f}dB, Voix faibles: {weak_voice_ratio*100:.1f}%")
        
        # Ajustement CONSERVATEUR de la durée minimale
        if recording_quality == 'low' or snr_estimate < 10:
            # Audio de mauvaise qualité : segments légèrement plus longs pour éviter la sur-segmentation
            optimized_params['min_duration_on'] = min(0.8, base_params['min_duration_on'] * 1.3)
            optimized_params['min_duration_off'] = min(0.2, base_params['min_duration_off'] * 1.5)
            if logger:
                logger.info("🔧 Ajustement pour audio de faible qualité: segments plus longs")
                
        elif recording_quality == 'high' and snr_estimate > 20:
            # Audio de haute qualité : segments légèrement plus courts pour plus de précision
            optimized_params['min_duration_on'] = max(0.3, base_params['min_duration_on'] * 0.8)
            optimized_params['min_duration_off'] = max(0.05, base_params['min_duration_off'] * 0.8)
            if logger:
                logger.info("🔧 Ajustement pour audio de haute qualité: segments plus précis")
        
        # Ajustement CONSERVATEUR des seuils pour voix faibles
        if weak_voice_ratio > 0.5:  # Beaucoup de voix faibles
            # Seuils légèrement plus sensibles
            optimized_params['onset'] = max(0.3, base_params['onset'] * 0.9)
            optimized_params['offset'] = max(0.3, base_params['offset'] * 0.9)
            if logger:
                logger.info("🔧 Ajustement pour voix faibles: seuils plus sensibles")
        
        if logger:
            logger.info("🎯 Paramètres optimisés basés sur l'analyse audio")
            
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Erreur optimisation paramètres: {e}, utilisation des paramètres par défaut")
        optimized_params = base_params
    
    return optimized_params

def diarize_audio_simple(audio_file: str, output_file: str = None, logger=None):
    """
    Diarisation simple et directe avec optimisation optionnelle.
    
    Utilise les paramètres par défaut optimisés de pyannote avec
    des ajustements légers basés sur l'analyse audio si disponible.
    """
    if not PYANNOTE_AVAILABLE:
        if logger:
            logger.error("❌ pyannote.audio non disponible")
        return False
    
    if not os.path.exists(audio_file):
        if logger:
            logger.error(f"❌ Fichier {audio_file} non trouvé")
        return False
    
    # Génération automatique du nom de sortie
    if not output_file:
        input_path = Path(audio_file)
        output_file = f"{input_path.stem}_diarization.txt"
    
    if logger:
        logger.info("============================================================")
        logger.info("🎭 DIARISATION SIMPLE ET EFFICACE")
        logger.info("============================================================")
        logger.info(f"📁 Fichier d'entrée: {audio_file}")
        logger.info(f"📁 Fichier de sortie: {output_file}")
    
    # Chargement de l'analyse audio du pré-traitement
    analysis = load_audio_analysis(audio_file, logger)
    
    # Optimisation des paramètres basée sur l'analyse
    params = optimize_diarization_parameters(analysis, logger)
    
    if logger:
        logger.info("🚀 Début de la diarisation...")
    
    start_time = time.time()
    
    try:
        # Chargement du pipeline pré-entraîné avec paramètres optimisés
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        # Application des paramètres optimisés (si différents des défauts)
        if params['clustering_threshold'] is not None:
            if hasattr(pipeline, '_clustering'):
                pipeline._clustering.threshold = params['clustering_threshold']
        
        if hasattr(pipeline, '_segmentation'):
            if params['min_duration_on'] != 0.5:
                pipeline._segmentation.min_duration_on = params['min_duration_on']
            if params['min_duration_off'] != 0.1:
                pipeline._segmentation.min_duration_off = params['min_duration_off']
            if params['onset'] != 0.5:
                pipeline._segmentation.onset = params['onset']
            if params['offset'] != 0.5:
                pipeline._segmentation.offset = params['offset']
        
        # Application directe sur tout le fichier
        diarization = pipeline(audio_file)
        
        # Extraction des informations
        speakers = set()
        segments = []
        total_speech_time = 0.0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start,
                'speaker': speaker
            })
            total_speech_time += turn.end - turn.start
        
        # Tri par temps de début
        segments.sort(key=lambda x: x['start'])
        
        # Sauvegarde des résultats
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DIARISATION DES LOCUTEURS\n")
            f.write("=" * 50 + "\n\n")
            
            if analysis:
                f.write("OPTIMISATION BASÉE SUR L'ANALYSE AUDIO:\n")
                f.write(f"• Qualité d'enregistrement: {analysis.get('recording_quality', 'N/A')}\n")
                f.write(f"• SNR estimé: {analysis.get('snr_estimate', 'N/A'):.1f} dB\n")
                f.write(f"• Ratio voix faibles: {analysis.get('weak_voice_ratio', 'N/A')*100:.1f}%\n")
                f.write("\nPARAMÈTRES APPLIQUÉS:\n")
                f.write(f"• Durée min. parole: {params['min_duration_on']:.2f}s\n")
                f.write(f"• Durée min. silence: {params['min_duration_off']:.2f}s\n")
                f.write(f"• Seuil début: {params['onset']:.2f}\n")
                f.write(f"• Seuil fin: {params['offset']:.2f}\n\n")
            
            f.write(f"Nombre de locuteurs détectés: {len(speakers)}\n")
            f.write(f"Nombre de segments: {len(segments)}\n")
            f.write(f"Temps de parole total: {total_speech_time:.1f}s\n\n")
            
            f.write("SEGMENTS PAR LOCUTEUR:\n")
            f.write("-" * 30 + "\n")
            
            for speaker in sorted(speakers):
                speaker_segments = [s for s in segments if s['speaker'] == speaker]
                speaker_time = sum(s['duration'] for s in speaker_segments)
                f.write(f"\n{speaker}: {len(speaker_segments)} segments, {speaker_time:.1f}s\n")
                
                for seg in speaker_segments:
                    f.write(f"  {seg['start']:7.1f}s - {seg['end']:7.1f}s ({seg['duration']:5.1f}s)\n")
            
            f.write("\nCHRONOLOGIE COMPLÈTE:\n")
            f.write("-" * 25 + "\n")
            
            for seg in segments:
                f.write(f"{seg['start']:7.1f}s - {seg['end']:7.1f}s | {seg['speaker']} ({seg['duration']:5.1f}s)\n")
        
        processing_time = time.time() - start_time
        
        if logger:
            logger.info("🎉 DIARISATION TERMINÉE")
            logger.info("-" * 30)
            logger.info(f"🎭 Locuteurs détectés: {len(speakers)}")
            logger.info(f"📊 Segments totaux: {len(segments)}")
            logger.info(f"🗣️ Temps de parole: {total_speech_time:.1f}s")
            logger.info(f"⏱️ Temps de traitement: {processing_time:.1f}s")
            logger.info(f"💾 Résultats sauvés: {output_file}")
            
            # Statistiques par locuteur
            for speaker in sorted(speakers):
                speaker_segments = [s for s in segments if s['speaker'] == speaker]
                speaker_time = sum(s['duration'] for s in speaker_segments)
                percentage = (speaker_time / total_speech_time) * 100
                logger.info(f"   {speaker}: {speaker_time:.1f}s ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Erreur lors de la diarisation: {str(e)}")
        return False

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Diarisation simple des locuteurs avec optimisation optionnelle")
    parser.add_argument("audio_file", help="Fichier audio à traiter")
    parser.add_argument("-o", "--output", help="Fichier de sortie (optionnel)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    success = diarize_audio_simple(
        audio_file=args.audio_file,
        output_file=args.output,
        logger=logger
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 