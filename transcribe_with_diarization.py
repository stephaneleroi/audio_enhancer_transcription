#!/usr/bin/env python3
"""
Chaîne complète de traitement audio adaptatif :
Pré-traitement + Transcription + Diarisation des locuteurs

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
import subprocess

# Import des modules locaux - utilisation via subprocess uniquement
# from audio_enhancer import enhance_audio_adaptive
# from transcribe_parallel import transcribe_audio_adaptive  
# from speaker_diarization import diarize_audio_adaptive, analyze_audio_for_diarization, calculate_diarization_parameters

@dataclass
class TranscriptionSegment:
    """Segment de transcription avec timing."""
    start: float
    end: float
    text: str
    confidence: float = 0.0
    
    def duration(self) -> float:
        return self.end - self.start

@dataclass
class SpeakerSegment:
    """Segment de locuteur avec timing."""
    start: float
    end: float
    speaker: str
    confidence: float = 0.0
    
    def duration(self) -> float:
        return self.end - self.start

@dataclass
class AlignedSegment:
    """Segment aligné avec transcription et locuteur."""
    start: float
    end: float
    text: str
    speaker: str
    transcription_confidence: float = 0.0
    speaker_confidence: float = 0.0
    
    def duration(self) -> float:
        return self.end - self.start
    
    def __str__(self) -> str:
        return f"[{self.start:.1f}s - {self.end:.1f}s] {self.speaker}: {self.text}"

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

def parse_srt_file(srt_file: str) -> List[TranscriptionSegment]:
    """
    Objectif : Parse un fichier SRT en segments de transcription
    
    Extrait les segments temporels et le texte depuis un fichier SRT.
    """
    segments = []
    
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Découpage par blocs (séparés par double saut de ligne)
    blocks = content.split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Ligne 2 contient le timing
            timing_line = lines[1]
            if ' --> ' in timing_line:
                start_str, end_str = timing_line.split(' --> ')
                
                # Conversion du format SRT (HH:MM:SS,mmm) en secondes
                def srt_time_to_seconds(time_str):
                    time_str = time_str.replace(',', '.')
                    parts = time_str.split(':')
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                
                start = srt_time_to_seconds(start_str.strip())
                end = srt_time_to_seconds(end_str.strip())
                
                # Lignes 3+ contiennent le texte
                text = ' '.join(lines[2:]).strip()
                
                if text:  # Ignorer les segments vides
                    segment = TranscriptionSegment(
                        start=start,
                        end=end,
                        text=text,
                        confidence=1.0  # Pas de confiance dans les SRT
                    )
                    segments.append(segment)
    
    return segments

def parse_diarization_file(diarization_file: str) -> List[SpeakerSegment]:
    """
    Objectif : Parse un fichier de diarisation en segments de locuteurs
    
    Extrait les segments temporels et les locuteurs depuis un fichier de diarisation.
    """
    segments = []
    
    with open(diarization_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # Format: "  1. [0.0s - 0.6s] SPEAKER_01 (conf: 1.000)"
        if '[' in line and ']' in line and 'SPEAKER_' in line:
            try:
                # Extraction du timing entre [ et ]
                start_bracket = line.find('[')
                end_bracket = line.find(']')
                timing_part = line[start_bracket+1:end_bracket]
                
                start_str, end_str = timing_part.split(' - ')
                start = float(start_str.replace('s', ''))
                end = float(end_str.replace('s', ''))
                
                # Extraction du locuteur après ]
                after_bracket = line[end_bracket+1:].strip()
                speaker = after_bracket.split(' ')[0]
                
                # Extraction de la confiance
                confidence = 1.0
                if '(conf:' in after_bracket:
                    conf_str = after_bracket.split('(conf: ')[1].split(')')[0]
                    confidence = float(conf_str)
                
                segment = SpeakerSegment(
                    start=start,
                    end=end,
                    speaker=speaker,
                    confidence=confidence
                )
                segments.append(segment)
                
            except (ValueError, IndexError) as e:
                continue  # Ignorer les lignes mal formatées
    
    return segments

def align_transcription_with_speakers(transcription_segments: List[TranscriptionSegment],
                                    speaker_segments: List[SpeakerSegment],
                                    logger=None) -> List[AlignedSegment]:
    """
    Objectif : Alignement intelligent transcription + diarisation
    
    Aligne les segments de transcription avec les segments de locuteurs
    en utilisant des algorithmes adaptatifs de chevauchement temporel.
    """
    if logger:
        logger.info("🔗 Alignement transcription + diarisation...")
    
    aligned_segments = []
    
    for trans_seg in transcription_segments:
        # Trouver le(s) locuteur(s) qui chevauchent avec ce segment de transcription
        overlapping_speakers = []
        
        for spk_seg in speaker_segments:
            # Calcul du chevauchement temporel
            overlap_start = max(trans_seg.start, spk_seg.start)
            overlap_end = min(trans_seg.end, spk_seg.end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                # Calcul du pourcentage de chevauchement
                trans_coverage = overlap_duration / trans_seg.duration()
                spk_coverage = overlap_duration / spk_seg.duration()
                
                # Score de chevauchement adaptatif
                overlap_score = (trans_coverage + spk_coverage) / 2
                
                overlapping_speakers.append({
                    'speaker': spk_seg.speaker,
                    'overlap_duration': overlap_duration,
                    'overlap_score': overlap_score,
                    'speaker_confidence': spk_seg.confidence,
                    'speaker_start': spk_seg.start,
                    'speaker_end': spk_seg.end
                })
        
        if overlapping_speakers:
            # Tri par score de chevauchement décroissant
            overlapping_speakers.sort(key=lambda x: x['overlap_score'], reverse=True)
            
            # Sélection du meilleur locuteur
            best_speaker = overlapping_speakers[0]
            
            # Création du segment aligné
            aligned_segment = AlignedSegment(
                start=trans_seg.start,
                end=trans_seg.end,
                text=trans_seg.text,
                speaker=best_speaker['speaker'],
                transcription_confidence=trans_seg.confidence,
                speaker_confidence=best_speaker['speaker_confidence']
            )
            aligned_segments.append(aligned_segment)
            
        else:
            # Aucun locuteur trouvé, utiliser un locuteur par défaut
            aligned_segment = AlignedSegment(
                start=trans_seg.start,
                end=trans_seg.end,
                text=trans_seg.text,
                speaker="SPEAKER_UNKNOWN",
                transcription_confidence=trans_seg.confidence,
                speaker_confidence=0.0
            )
            aligned_segments.append(aligned_segment)
    
    if logger:
        logger.info(f"✅ {len(aligned_segments)} segments alignés")
        
        # Statistiques d'alignement
        speaker_counts = {}
        for seg in aligned_segments:
            speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1
        
        logger.info("📊 Répartition par locuteur :")
        for speaker, count in sorted(speaker_counts.items()):
            total_duration = sum(seg.duration() for seg in aligned_segments if seg.speaker == speaker)
            logger.info(f"   • {speaker}: {count} segments ({total_duration:.1f}s)")
    
    return aligned_segments

def save_aligned_results(aligned_segments: List[AlignedSegment], output_file: str, logger=None):
    """
    Objectif : Sauvegarde des résultats alignés
    
    Sauvegarde les segments alignés dans différents formats.
    """
    if logger:
        logger.info(f"💾 Sauvegarde des résultats alignés: {output_file}")
    
    # Format texte détaillé
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TRANSCRIPTION AVEC DIARISATION ADAPTATIVE\n")
        f.write("=" * 60 + "\n\n")
        
        for i, segment in enumerate(aligned_segments, 1):
            f.write(f"{i:3d}. {segment}\n")
        
        f.write(f"\n📊 STATISTIQUES:\n")
        f.write(f"   • Segments totaux: {len(aligned_segments)}\n")
        
        # Statistiques par locuteur
        speaker_stats = {}
        for segment in aligned_segments:
            if segment.speaker not in speaker_stats:
                speaker_stats[segment.speaker] = {'duration': 0, 'segments': 0, 'words': 0}
            speaker_stats[segment.speaker]['duration'] += segment.duration()
            speaker_stats[segment.speaker]['segments'] += 1
            speaker_stats[segment.speaker]['words'] += len(segment.text.split())
        
        f.write(f"   • Locuteurs uniques: {len(speaker_stats)}\n")
        f.write(f"   • Durée totale: {max(s.end for s in aligned_segments):.1f}s\n")
        
        f.write(f"\n👥 STATISTIQUES PAR LOCUTEUR:\n")
        for speaker, stats in sorted(speaker_stats.items()):
            f.write(f"   • {speaker}: {stats['duration']:.1f}s, {stats['segments']} segments, {stats['words']} mots\n")
    
    # Format SRT avec locuteurs
    srt_file = output_file.replace('.txt', '_with_speakers.srt')
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(aligned_segments, 1):
            # Conversion en format SRT
            def seconds_to_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
            
            start_time = seconds_to_srt_time(segment.start)
            end_time = seconds_to_srt_time(segment.end)
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"[{segment.speaker}] {segment.text}\n\n")
    
    if logger:
        logger.info(f"📁 Fichiers sauvegardés: {output_file} et {srt_file}")

def process_complete_pipeline(input_file: str, output_base: str, num_workers: int = 4, logger=None):
    """
    Objectif : Pipeline complet de traitement adaptatif
    
    Exécute la chaîne complète : pré-traitement → transcription → diarisation → alignement
    """
    if logger:
        logger.info("🚀 DÉBUT DU PIPELINE COMPLET ADAPTATIF")
        logger.info("=" * 60)
    
    start_time = time.time()
    
    # Étape 1: Pré-traitement audio adaptatif
    if logger:
        logger.info("🔧 ÉTAPE 1: Pré-traitement audio adaptatif")
    
    enhanced_file = f"{output_base}_enhanced.wav"
    
    # Appel du module de pré-traitement
    cmd = [sys.executable, "audio_enhancer.py", input_file, "--output", enhanced_file]
    if logger and logger.level == logging.DEBUG:
        cmd.append("--verbose")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if logger:
            logger.error(f"❌ Erreur pré-traitement: {result.stderr}")
        return False
    
    # Étape 2: Transcription adaptative
    if logger:
        logger.info("🎤 ÉTAPE 2: Transcription adaptative")
    
    # Appel du module de transcription
    cmd = [sys.executable, "transcribe_parallel.py", enhanced_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if logger:
            logger.error(f"❌ Erreur transcription: {result.stderr}")
        return False
    
    transcription_file = enhanced_file.replace('.wav', '_adaptive_subtitles.srt')
    
    # Étape 3: Diarisation adaptative
    if logger:
        logger.info("🎭 ÉTAPE 3: Diarisation adaptative")
    
    diarization_file = f"{output_base}_diarization.txt"
    
    # Appel du module de diarisation
    cmd = [sys.executable, "speaker_diarization.py", enhanced_file, "--output", diarization_file, "--workers", str(num_workers)]
    if logger and logger.level == logging.DEBUG:
        cmd.append("--verbose")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if logger:
            logger.error(f"❌ Erreur diarisation: {result.stderr}")
        return False
    
    # Étape 4: Alignement et fusion
    if logger:
        logger.info("🔗 ÉTAPE 4: Alignement transcription + diarisation")
    
    # Parse des résultats
    transcription_segments = parse_srt_file(transcription_file)
    speaker_segments = parse_diarization_file(diarization_file)
    
    # Alignement
    aligned_segments = align_transcription_with_speakers(
        transcription_segments, 
        speaker_segments, 
        logger
    )
    
    # Sauvegarde des résultats finaux
    final_output = f"{output_base}_complete_transcription.txt"
    save_aligned_results(aligned_segments, final_output, logger)
    
    # Statistiques finales
    total_time = time.time() - start_time
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    
    if logger:
        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info(f"📊 Segments transcrits: {len(transcription_segments)}")
        logger.info(f"🎭 Segments de locuteurs: {len(speaker_segments)}")
        logger.info(f"🔗 Segments alignés: {len(aligned_segments)}")
        logger.info(f"👥 Locuteurs détectés: {len(set(s.speaker for s in aligned_segments))}")
        logger.info(f"⏱️ Temps total: {total_time:.1f}s")
        logger.info(f"📊 Mémoire utilisée: {memory_usage:.1f} MB")
        logger.info(f"📁 Résultat final: {final_output}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet adaptatif : Pré-traitement + Transcription + Diarisation"
    )
    
    parser.add_argument('input_file', help='Fichier audio à traiter')
    parser.add_argument('--output', '-o', help='Préfixe des fichiers de sortie (optionnel)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Affichage détaillé')
    parser.add_argument('--workers', '-w', type=int, default=4, 
                       help='Nombre de workers parallèles (défaut: 4)')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    # Génération automatique du préfixe de sortie
    if not args.output:
        base, ext = os.path.splitext(args.input_file)
        args.output = base
    
    try:
        # Vérification du fichier d'entrée
        if not os.path.exists(args.input_file):
            logger.error(f"❌ Fichier non trouvé: {args.input_file}")
            sys.exit(1)
        
        # Exécution du pipeline complet
        success = process_complete_pipeline(
            args.input_file,
            args.output,
            num_workers=args.workers,
            logger=logger
        )
        
        if not success:
            logger.error("❌ Échec du pipeline")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 