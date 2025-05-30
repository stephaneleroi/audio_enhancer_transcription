#!/usr/bin/env python3
"""
Pipeline complet optimisé : Pré-traitement → Transcription → Diarisation → Alignement
Avec optimisation intelligente basée sur l'analyse audio partagée
"""

import os
import sys
import argparse
import logging
import time
import subprocess
import json
from pathlib import Path

def setup_logging(verbose=False):
    """Configure le système de journalisation."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Supprimer les handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def run_audio_enhancement(input_file: str, logger=None):
    """
    Étape 1: Pré-traitement audio adaptatif avec sauvegarde de l'analyse.
    
    Cette étape améliore la qualité audio ET sauvegarde l'analyse
    pour optimiser les étapes suivantes (transcription et diarisation).
    """
    if logger:
        logger.info("🔧 ÉTAPE 1: Pré-traitement audio adaptatif")
    
    input_path = Path(input_file)
    enhanced_file = input_path.with_name(f"{input_path.stem}_adaptive_enhanced{input_path.suffix}")
    
    # Commande avec sauvegarde de l'analyse
    cmd = [
        sys.executable, "audio_enhancer.py",
        str(input_file),
        "--output", str(enhanced_file),
        "--save-analysis",  # NOUVEAU: Sauvegarde l'analyse pour les étapes suivantes
        "--verbose" if logger and logger.level == logging.DEBUG else ""
    ]
    
    # Filtrer les arguments vides
    cmd = [arg for arg in cmd if arg]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if logger and logger.level == logging.DEBUG:
            logger.debug(f"Audio enhancer output: {result.stdout}")
        
        return str(enhanced_file)
        
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"❌ Erreur pré-traitement: {e}")
            logger.error(f"Stderr: {e.stderr}")
        return None

def run_adaptive_transcription(audio_file: str, logger=None):
    """
    Étape 2: Transcription adaptative parallèle.
    
    Utilise le script de transcription parallèle adaptatif qui peut bénéficier
    de l'analyse audio sauvegardée lors du pré-traitement.
    """
    if logger:
        logger.info("🎤 ÉTAPE 2: Transcription adaptative parallèle")
    
    cmd = [
        sys.executable, "transcribe_parallel.py",
        audio_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if logger and logger.level == logging.DEBUG:
            logger.debug(f"Transcription output: {result.stdout}")
        
        # Le script génère automatiquement le fichier de transcription
        audio_path = Path(audio_file)
        transcription_file = audio_path.with_name(f"{audio_path.stem}_adaptive_transcription.txt")
        
        if transcription_file.exists():
            return str(transcription_file)
        else:
            if logger:
                logger.error(f"❌ Fichier de transcription non trouvé: {transcription_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"❌ Erreur transcription: {e}")
            logger.error(f"Stderr: {e.stderr}")
        return None

def run_optimized_diarization(audio_file: str, logger=None):
    """
    Étape 3: Diarisation optimisée basée sur l'analyse audio.
    
    Utilise le script de diarisation amélioré qui charge automatiquement
    l'analyse audio sauvegardée lors du pré-traitement pour optimiser
    ses paramètres.
    """
    if logger:
        logger.info("🎭 ÉTAPE 3: Diarisation optimisée")
    
    audio_path = Path(audio_file)
    diarization_file = audio_path.with_name(f"{audio_path.stem}_diarization.txt")
    
    cmd = [
        sys.executable, "speaker_diarization_simple.py",
        audio_file,
        "--output", str(diarization_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if logger and logger.level == logging.DEBUG:
            logger.debug(f"Diarization output: {result.stdout}")
        
        if diarization_file.exists():
            return str(diarization_file)
        else:
            if logger:
                logger.error(f"❌ Fichier de diarisation non trouvé: {diarization_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"❌ Erreur diarisation: {e}")
            logger.error(f"Stderr: {e.stderr}")
        return None

def parse_transcription_file(transcription_file: str):
    """Parse le fichier de transcription pour extraire les segments."""
    segments = []
    
    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Recherche des segments dans le format du transcripteur adaptatif
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Format: "  1. [0m11s - 0m19s]  Texte..."
            if '[' in line and ']' in line and 'm' in line and 's' in line:
                try:
                    # Extraire la partie entre crochets
                    start_bracket = line.find('[')
                    end_bracket = line.find(']')
                    
                    if start_bracket != -1 and end_bracket != -1:
                        time_part = line[start_bracket+1:end_bracket]
                        text_part = line[end_bracket+1:].strip()
                        
                        # Parser le timing "0m11s - 0m19s"
                        if ' - ' in time_part:
                            start_str, end_str = time_part.split(' - ')
                            
                            # Convertir "0m11s" en secondes
                            def time_to_seconds(time_str):
                                time_str = time_str.strip()
                                seconds = 0
                                
                                if 'h' in time_str:
                                    parts = time_str.split('h')
                                    seconds += int(parts[0]) * 3600
                                    time_str = parts[1] if len(parts) > 1 else '0m0s'
                                
                                if 'm' in time_str:
                                    parts = time_str.split('m')
                                    seconds += int(parts[0]) * 60
                                    time_str = parts[1] if len(parts) > 1 else '0s'
                                
                                if 's' in time_str:
                                    seconds += int(time_str.replace('s', ''))
                                
                                return seconds
                            
                            start = time_to_seconds(start_str)
                            end = time_to_seconds(end_str)
                            
                            if text_part:  # Ignorer les segments vides
                                segments.append({
                                    'start': start,
                                    'end': end,
                                    'text': text_part
                                })
                except Exception as e:
                    print(f"Erreur parsing ligne: {line} - {e}")
                    continue
        
        return segments
        
    except Exception as e:
        print(f"Erreur parsing transcription: {e}")
        return []

def parse_diarization_file(diarization_file: str):
    """Parse le fichier de diarisation pour extraire les segments de locuteurs."""
    speaker_segments = []
    
    try:
        with open(diarization_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Recherche de la section chronologie
        lines = content.split('\n')
        in_chronology = False
        
        for line in lines:
            line = line.strip()
            
            if "CHRONOLOGIE COMPLÈTE:" in line:
                in_chronology = True
                continue
            
            if in_chronology and line and not line.startswith('-'):
                try:
                    # Format: "11.0s - 13.5s | SPEAKER_00 (2.5s)"
                    if ' | ' in line:
                        time_part, speaker_part = line.split(' | ')
                        start_str, end_str = time_part.split(' - ')
                        
                        start = float(start_str.replace('s', ''))
                        end = float(end_str.replace('s', ''))
                        
                        speaker = speaker_part.split(' ')[0]
                        
                        speaker_segments.append({
                            'start': start,
                            'end': end,
                            'speaker': speaker
                        })
                except:
                    continue
        
        return speaker_segments
        
    except Exception as e:
        print(f"Erreur parsing diarisation: {e}")
        return []

def align_transcription_with_speakers(transcription_segments, speaker_segments, logger=None):
    """
    Aligne les segments de transcription avec les locuteurs détectés.
    
    Utilise une approche de chevauchement temporel pour associer
    chaque segment de transcription au locuteur le plus probable.
    """
    if logger:
        logger.info("🔗 Alignement transcription + diarisation...")
    
    aligned_segments = []
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg['start']
        trans_end = trans_seg['end']
        trans_center = (trans_start + trans_end) / 2
        
        # Trouver le locuteur avec le plus grand chevauchement
        best_speaker = "UNKNOWN"
        best_overlap = 0
        
        for spk_seg in speaker_segments:
            spk_start = spk_seg['start']
            spk_end = spk_seg['end']
            
            # Calcul du chevauchement
            overlap_start = max(trans_start, spk_start)
            overlap_end = min(trans_end, spk_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Normalisation par la durée du segment de transcription
            trans_duration = trans_end - trans_start
            overlap_ratio = overlap_duration / trans_duration if trans_duration > 0 else 0
            
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = spk_seg['speaker']
        
        # Si aucun chevauchement significatif, utiliser le locuteur le plus proche
        if best_overlap < 0.1:
            min_distance = float('inf')
            for spk_seg in speaker_segments:
                spk_center = (spk_seg['start'] + spk_seg['end']) / 2
                distance = abs(trans_center - spk_center)
                if distance < min_distance:
                    min_distance = distance
                    best_speaker = spk_seg['speaker']
        
        aligned_segments.append({
            'start': trans_start,
            'end': trans_end,
            'text': trans_seg['text'],
            'speaker': best_speaker,
            'confidence': best_overlap
        })
    
    if logger:
        logger.info(f"✅ {len(aligned_segments)} segments alignés")
    
    return aligned_segments

def save_aligned_results(aligned_segments, output_base: str, logger=None):
    """Sauvegarde les résultats alignés en format texte et SRT."""
    
    # Fichier texte
    txt_file = f"{output_base}_complete_transcription.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("TRANSCRIPTION COMPLÈTE AVEC LOCUTEURS\n")
        f.write("=" * 50 + "\n\n")
        
        for seg in aligned_segments:
            f.write(f"{seg['start']:7.1f}s - {seg['end']:7.1f}s | {seg['speaker']}\n")
            f.write(f"  {seg['text']}\n\n")
    
    # Fichier SRT
    srt_file = f"{output_base}_complete_transcription_with_speakers.srt"
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(aligned_segments, 1):
            start_time = format_srt_time(seg['start'])
            end_time = format_srt_time(seg['end'])
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"[{seg['speaker']}] {seg['text']}\n\n")
    
    if logger:
        logger.info(f"💾 Sauvegarde des résultats alignés: {txt_file}")
        logger.info(f"📁 Fichiers sauvegardés: {txt_file} et {srt_file}")
    
    return txt_file, srt_file

def format_srt_time(seconds):
    """Formate le temps en format SRT (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def get_speaker_statistics(aligned_segments):
    """Calcule les statistiques par locuteur."""
    speaker_stats = {}
    
    for seg in aligned_segments:
        speaker = seg['speaker']
        duration = seg['end'] - seg['start']
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'segments': 0,
                'total_time': 0.0
            }
        
        speaker_stats[speaker]['segments'] += 1
        speaker_stats[speaker]['total_time'] += duration
    
    return speaker_stats

def main():
    """Pipeline complet optimisé."""
    parser = argparse.ArgumentParser(description="Pipeline complet avec optimisation basée sur l'analyse audio")
    parser.add_argument("input_file", help="Fichier audio à traiter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument("--skip-enhancement", action="store_true", help="Ignorer le pré-traitement")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    start_time = time.time()
    
    logger.info("🚀 DÉBUT DU PIPELINE COMPLET OPTIMISÉ")
    logger.info("=" * 60)
    
    try:
        # Étape 1: Pré-traitement audio avec sauvegarde de l'analyse
        if not args.skip_enhancement:
            enhanced_file = run_audio_enhancement(args.input_file, logger)
            if not enhanced_file:
                logger.error("❌ Échec du pré-traitement")
                return 1
            audio_file = enhanced_file
        else:
            audio_file = args.input_file
            logger.info("⏭️ Pré-traitement ignoré")
        
        # Étape 2: Transcription adaptative
        transcription_file = run_adaptive_transcription(audio_file, logger)
        if not transcription_file:
            logger.error("❌ Échec de la transcription")
            return 1
        
        # Étape 3: Diarisation optimisée (utilise automatiquement l'analyse sauvegardée)
        diarization_file = run_optimized_diarization(audio_file, logger)
        if not diarization_file:
            logger.error("❌ Échec de la diarisation")
            return 1
        
        # Étape 4: Alignement
        logger.info("🔗 ÉTAPE 4: Alignement transcription + diarisation")
        
        transcription_segments = parse_transcription_file(transcription_file)
        speaker_segments = parse_diarization_file(diarization_file)
        
        if not transcription_segments:
            logger.error("❌ Aucun segment de transcription trouvé")
            return 1
        
        if not speaker_segments:
            logger.error("❌ Aucun segment de locuteur trouvé")
            return 1
        
        aligned_segments = align_transcription_with_speakers(
            transcription_segments, speaker_segments, logger
        )
        
        # Sauvegarde des résultats
        input_path = Path(args.input_file)
        output_base = input_path.stem
        
        txt_file, srt_file = save_aligned_results(aligned_segments, output_base, logger)
        
        # Statistiques finales
        speaker_stats = get_speaker_statistics(aligned_segments)
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info(f"📊 Segments transcrits: {len(transcription_segments)}")
        logger.info(f"🎭 Segments de locuteurs: {len(speaker_segments)}")
        logger.info(f"🔗 Segments alignés: {len(aligned_segments)}")
        logger.info(f"👥 Locuteurs détectés: {len(speaker_stats)}")
        
        logger.info("📊 Répartition par locuteur :")
        for speaker, stats in speaker_stats.items():
            logger.info(f"   • {speaker}: {stats['segments']} segments ({stats['total_time']:.1f}s)")
        
        logger.info(f"⏱️ Temps total: {total_time:.1f}s")
        
        # Mémoire utilisée
        try:
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"📊 Mémoire utilisée: {memory_usage:.1f} MB")
        except ImportError:
            pass
        
        logger.info(f"📁 Résultat final: {txt_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 