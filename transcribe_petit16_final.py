#!/usr/bin/env python3
"""
Script pour re-transcrire petit16_enhanced.wav avec la méthode deux phases
pour comparaison directe avec gros16_enhanced.wav
"""

import sys
import os
import time
import numpy as np
sys.path.append('.')

# Import du module adaptatif
import importlib.util
spec = importlib.util.spec_from_file_location("adaptive_vad_transcription", "adaptive_vad_transcription.py.py")
adaptive_vad_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive_vad_module)

# Import des fonctions
test_multiple_vad_configs = adaptive_vad_module.test_multiple_vad_configs
analyze_audio_characteristics = adaptive_vad_module.analyze_audio_characteristics
from faster_whisper import WhisperModel

def format_time(seconds):
    """Format timestamp pour lecture humaine"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    else:
        return f"{minutes}m{secs:02d}s"

def transcribe_petit16_detailed():
    """
    Transcription détaillée de petit16_enhanced.wav avec affichage segment par segment
    """
    
    audio_file = "petit16_enhanced.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Fichier non trouvé: {audio_file}")
        return None
    
    print("🎯 TRANSCRIPTION DÉTAILLÉE PETIT16_ENHANCED.WAV")
    print("="*60)
    
    start_time = time.time()
    
    # Test des configurations sur le fichier complet
    print("🔬 CALIBRATION VAD SUR FICHIER COMPLET...")
    print("-" * 50)
    
    try:
        calibration_start = time.time()
        
        best_vad_params, segments, transcription_config = test_multiple_vad_configs(
            audio_file,
            content_type="meeting"
        )
        
        calibration_time = time.time() - calibration_start
        
        print(f"✅ Calibration terminée en {calibration_time:.1f}s")
        print(f"🎯 Paramètres optimaux trouvés:")
        for key, value in best_vad_params.items():
            print(f"   {key}: {value}")
        
        if segments:
            confidence = np.mean([s.avg_logprob for s in segments])
            print(f"🎭 Confiance: {confidence:.3f}")
            print(f"📝 Segments: {len(segments)}")
            
            print(f"\n📝 TRANSCRIPTION COMPLÈTE PETIT16:")
            print("=" * 60)
            
            for i, segment in enumerate(segments, 1):
                timestamp_start = format_time(segment.start)
                timestamp_end = format_time(segment.end)
                print(f"{i:2d}. [{timestamp_start} - {timestamp_end}] (conf: {segment.avg_logprob:.3f})")
                print(f"    {segment.text}")
                print()
            
            print("=" * 60)
            
            # Sauvegarde pour comparaison
            output_file = "petit16_enhanced_detailed_transcription.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# TRANSCRIPTION DÉTAILLÉE - {audio_file}\n")
                f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Segments: {len(segments)}\n")
                f.write(f"# Confiance moyenne: {confidence:.3f}\n")
                f.write(f"# Paramètres VAD optimaux: {best_vad_params}\n")
                f.write("="*60 + "\n\n")
                
                for segment in segments:
                    timestamp_start = format_time(segment.start)
                    timestamp_end = format_time(segment.end)
                    f.write(f"[{timestamp_start} - {timestamp_end}] {segment.text}\n")
            
            print(f"✅ Sauvegardé: {output_file}")
            
            return segments, best_vad_params, confidence
        
        else:
            print("❌ Aucun segment transcrit")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        return None, None, None

def compare_with_gros16_first_40s():
    """
    Compare les 40 premières secondes de gros16 avec petit16
    """
    print("\n🔍 COMPARAISON AVEC GROS16 (40 premières secondes)")
    print("=" * 60)
    
    # Lire la transcription de gros16 deux phases
    try:
        with open("gros16_enhanced_two_phase_transcription.txt", "r", encoding="utf-8") as f:
            gros16_content = f.read()
        
        # Extraire les segments des 40 premières secondes
        gros16_40s_segments = []
        lines = gros16_content.split('\n')
        
        for line in lines:
            if line.startswith('[') and ']' in line:
                # Parser le timestamp
                timestamp_part = line.split(']')[0] + ']'
                text_part = line.split('] ', 1)[1] if '] ' in line else ''
                
                # Extraire le temps de début
                if 'h' in timestamp_part:
                    # Format avec heures
                    time_str = timestamp_part.split(' - ')[0][1:]  # Enlever le [
                    if 'h' in time_str and 'm' in time_str and 's' in time_str:
                        parts = time_str.replace('h', ':').replace('m', ':').replace('s', '').split(':')
                        start_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    else:
                        continue
                else:
                    # Format minutes:secondes
                    time_str = timestamp_part.split(' - ')[0][1:]  # Enlever le [
                    if 'm' in time_str and 's' in time_str:
                        parts = time_str.replace('m', ':').replace('s', '').split(':')
                        start_seconds = int(parts[0]) * 60 + int(parts[1])
                    else:
                        continue
                
                if start_seconds < 40:
                    gros16_40s_segments.append((start_seconds, text_part))
        
        print(f"📊 GROS16 (40s): {len(gros16_40s_segments)} segments")
        
        for i, (start_time, text) in enumerate(gros16_40s_segments, 1):
            print(f"  {i:2d}. [{start_time:2d}s] {text}")
        
        print(f"\n📋 RÉSUMÉ COMPARAISON:")
        print(f"   - Petit16 (40s total): 11 segments (référence)")
        print(f"   - Gros16 (40s premiers): {len(gros16_40s_segments)} segments")
        print(f"   - Différence: {len(gros16_40s_segments) - 11:+d} segments")
        
        if len(gros16_40s_segments) < 11:
            print(f"   ⚠️ Il manque {11 - len(gros16_40s_segments)} segments dans gros16!")
        
    except FileNotFoundError:
        print("❌ Fichier gros16_enhanced_two_phase_transcription.txt non trouvé")
    except Exception as e:
        print(f"❌ Erreur comparaison: {e}")

if __name__ == "__main__":
    print("🎯 ANALYSE DÉTAILLÉE PETIT16 + COMPARAISON")
    print("="*60)
    
    # Transcription détaillée de petit16
    segments, vad_params, confidence = transcribe_petit16_detailed()
    
    if segments:
        print(f"\n🎉 TRANSCRIPTION PETIT16 RÉUSSIE!")
        print(f"📝 {len(segments)} segments, confiance {confidence:.3f}")
        
        # Comparaison avec gros16
        compare_with_gros16_first_40s()
        
    else:
        print(f"\n❌ ÉCHEC TRANSCRIPTION PETIT16") 