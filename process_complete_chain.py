#!/usr/bin/env python3
"""
Chaîne de traitement complète pour un fichier audio :
1. Extraction audio en WAV
2. Amélioration audio
3. Transcription adaptative
4. Diarisation adaptative
5. Fusion en SRT final avec locuteurs et transcription
"""

import os
import sys
import argparse
import logging
import subprocess
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

def run_command(command, logger, description=""):
    """Exécute une commande et gère les erreurs."""
    if description:
        logger.info(f"🔄 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            logger.debug(f"Sortie: {result.stdout.strip()}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lors de l'exécution: {e}")
        if e.stderr:
            logger.error(f"Erreur détaillée: {e.stderr}")
        return False, e.stderr

def extract_audio_to_wav(input_file, logger):
    """Étape 1: Extraction audio en WAV."""
    base_name = Path(input_file).stem
    wav_file = f"{base_name}.wav"
    
    if os.path.exists(wav_file):
        logger.info(f"✅ Fichier WAV existe déjà: {wav_file}")
        return wav_file
    
    command = f"python extract_audio.py {input_file}"
    success, output = run_command(command, logger, f"Extraction audio de {input_file}")
    
    if success and os.path.exists(wav_file):
        logger.info(f"✅ Audio extrait: {wav_file}")
        return wav_file
    else:
        logger.error(f"❌ Échec de l'extraction audio")
        return None

def enhance_audio(wav_file, logger):
    """Étape 2: Amélioration audio."""
    base_name = Path(wav_file).stem
    enhanced_file = f"{base_name}_adaptive_enhanced.wav"
    
    if os.path.exists(enhanced_file):
        logger.info(f"✅ Fichier amélioré existe déjà: {enhanced_file}")
        return enhanced_file
    
    command = f"python audio_enhancer.py {wav_file}"
    success, output = run_command(command, logger, f"Amélioration audio de {wav_file}")
    
    if success and os.path.exists(enhanced_file):
        logger.info(f"✅ Audio amélioré: {enhanced_file}")
        return enhanced_file
    else:
        logger.error(f"❌ Échec de l'amélioration audio")
        return None

def transcribe_audio(enhanced_file, logger):
    """Étape 3: Transcription adaptative."""
    base_name = Path(enhanced_file).stem
    transcription_file = f"{base_name}_adaptive_subtitles.srt"
    
    if os.path.exists(transcription_file):
        logger.info(f"✅ Transcription existe déjà: {transcription_file}")
        return transcription_file
    
    command = f"python transcribe_parallel.py {enhanced_file}"
    success, output = run_command(command, logger, f"Transcription adaptative de {enhanced_file}")
    
    if success and os.path.exists(transcription_file):
        logger.info(f"✅ Transcription terminée: {transcription_file}")
        return transcription_file
    else:
        logger.error(f"❌ Échec de la transcription")
        return None

def diarize_audio(enhanced_file, logger):
    """Étape 4: Diarisation adaptative."""
    base_name = Path(enhanced_file).stem
    diarization_file = f"{base_name}_diarization.txt"
    
    if os.path.exists(diarization_file):
        logger.info(f"✅ Diarisation existe déjà: {diarization_file}")
        return diarization_file
    
    command = f"python speaker_diarization.py {enhanced_file}"
    success, output = run_command(command, logger, f"Diarisation adaptative de {enhanced_file}")
    
    if success and os.path.exists(diarization_file):
        logger.info(f"✅ Diarisation terminée: {diarization_file}")
        return diarization_file
    else:
        logger.error(f"❌ Échec de la diarisation")
        return None

def parse_srt_file(srt_file):
    """Parse un fichier SRT et retourne les segments."""
    segments = []
    
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    # Pattern pour matcher les blocs SRT
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\n*$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        start_time = match[1]
        end_time = match[2]
        text = match[3].strip()
        
        segments.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return segments

def parse_diarization_file(diarization_file):
    """Parse un fichier de diarisation et retourne les segments."""
    segments = []
    
    with open(diarization_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    lines = content.split('\n')
    in_segments_section = False
    
    for line in lines:
        if "SEGMENTS DÉTECTÉS:" in line:
            in_segments_section = True
            continue
        
        if in_segments_section and line.strip():
            # Pattern pour matcher: "  1. [0.0s - 0.9s] SPEAKER_04 (conf: 1.000)"
            match = re.search(r'\d+\.\s*\[(\d+\.?\d*)s\s*-\s*(\d+\.?\d*)s\]\s*(\w+)', line)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                speaker = match.group(3)
                
                segments.append({
                    'start': start,
                    'end': end,
                    'speaker': speaker
                })
            elif line.startswith('📊') or line.startswith('👥'):
                break
    
    return segments

def seconds_to_srt_time(seconds):
    """Convertit des secondes en format SRT (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def merge_transcription_and_diarization(transcription_file, diarization_file, output_file, logger):
    """Étape 5: Fusion transcription + diarisation en SRT final."""
    logger.info("🔗 Fusion transcription et diarisation...")
    
    # Parse des fichiers
    transcription_segments = parse_srt_file(transcription_file)
    diarization_segments = parse_diarization_file(diarization_file)
    
    logger.info(f"   • Segments de transcription: {len(transcription_segments)}")
    logger.info(f"   • Segments de diarisation: {len(diarization_segments)}")
    
    # Fonction pour convertir timestamp SRT en secondes
    def srt_time_to_seconds(time_str):
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    
    # Fusion des segments
    merged_segments = []
    
    for i, trans_seg in enumerate(transcription_segments, 1):
        trans_start = srt_time_to_seconds(trans_seg['start'])
        trans_end = srt_time_to_seconds(trans_seg['end'])
        trans_mid = (trans_start + trans_end) / 2
        
        # Trouver le locuteur correspondant
        best_speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0
        
        for diar_seg in diarization_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            
            # Calculer le chevauchement
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg['speaker']
        
        # Créer le segment fusionné
        merged_text = f"[{best_speaker}] {trans_seg['text']}"
        
        merged_segments.append({
            'index': i,
            'start': trans_seg['start'],
            'end': trans_seg['end'],
            'text': merged_text
        })
    
    # Écriture du fichier SRT final
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in merged_segments:
            f.write(f"{segment['index']}\n")
            f.write(f"{segment['start']} --> {segment['end']}\n")
            f.write(f"{segment['text']}\n\n")
    
    logger.info(f"✅ Fichier SRT final créé: {output_file}")
    logger.info(f"   • {len(merged_segments)} segments avec locuteurs identifiés")
    
    return output_file

def process_complete_chain(input_file, logger):
    """Traitement complet d'un fichier audio."""
    logger.info("=" * 60)
    logger.info("🚀 CHAÎNE DE TRAITEMENT COMPLÈTE")
    logger.info("=" * 60)
    logger.info(f"📁 Fichier d'entrée: {input_file}")
    
    # Étape 1: Extraction audio
    wav_file = extract_audio_to_wav(input_file, logger)
    if not wav_file:
        return False
    
    # Étape 2: Amélioration audio
    enhanced_file = enhance_audio(wav_file, logger)
    if not enhanced_file:
        return False
    
    # Étape 3: Transcription
    transcription_file = transcribe_audio(enhanced_file, logger)
    if not transcription_file:
        return False
    
    # Étape 4: Diarisation
    diarization_file = diarize_audio(enhanced_file, logger)
    if not diarization_file:
        return False
    
    # Étape 5: Fusion
    base_name = Path(input_file).stem
    final_srt = f"{base_name}_final_with_speakers.srt"
    
    merge_transcription_and_diarization(transcription_file, diarization_file, final_srt, logger)
    
    logger.info("=" * 60)
    logger.info("✅ CHAÎNE DE TRAITEMENT TERMINÉE AVEC SUCCÈS")
    logger.info("=" * 60)
    logger.info(f"📁 Fichier final: {final_srt}")
    
    return True

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Chaîne de traitement complète audio → transcription + diarisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Étapes de traitement :
1. Extraction audio en WAV
2. Amélioration audio
3. Transcription adaptative
4. Diarisation adaptative
5. Fusion en SRT final avec locuteurs

Exemple d'utilisation:
  python process_complete_chain.py petit.mp3
        """
    )
    
    parser.add_argument("input_file", help="Fichier audio d'entrée")
    parser.add_argument("--verbose", "-v", action="store_true", help="Affichage détaillé")
    
    args = parser.parse_args()
    
    # Configuration du logging
    logger = setup_logging(args.verbose)
    
    if not os.path.exists(args.input_file):
        logger.error(f"❌ Fichier non trouvé: {args.input_file}")
        sys.exit(1)
    
    try:
        success = process_complete_chain(args.input_file, logger)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement: {str(e)}")
        if args.verbose:
            import traceback
            logger.error("Traceback:")
            for line in traceback.format_exc().strip().split('\n'):
                logger.error(line)
        sys.exit(1)

if __name__ == "__main__":
    main() 