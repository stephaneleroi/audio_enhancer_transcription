#!/usr/bin/env python3
"""
Script final optimisé pour gros16_enhanced.wav
Utilise directement les paramètres VAD optimaux trouvés pour petit16
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

def transcribe_chunk_with_timing(audio_file, start_time, end_time, vad_params, config):
    """Transcrit un chunk spécifique avec timing précis"""
    try:
        import subprocess
        
        chunk_file = f"temp_chunk_{int(time.time())}_{start_time:.0f}s.wav"
        
        # Extraire le chunk avec ffmpeg
        cmd = [
            'ffmpeg', '-i', audio_file, 
            '-ss', str(start_time), 
            '-t', str(end_time - start_time),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-y', chunk_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    ⚠️ Erreur extraction chunk: {result.stderr}")
            return []
        
        # Transcription du chunk
        model = WhisperModel('large-v3', device='cpu', compute_type='int8')
        
        segments, info = model.transcribe(
            chunk_file,
            vad_parameters=vad_params,
            **config
        )
        
        segment_list = list(segments)
        
        # Nettoyage
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
        
        return segment_list
        
    except Exception as e:
        print(f"    ⚠️ Erreur transcription chunk: {e}")
        if 'chunk_file' in locals() and os.path.exists(chunk_file):
            os.remove(chunk_file)
        return []

def transcribe_gros16_final():
    """
    Transcription finale optimisée de gros16_enhanced.wav
    Utilise les paramètres VAD optimaux de petit16
    """
    
    audio_file = "gros16_enhanced.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Fichier non trouvé: {audio_file}")
        return None
    
    print("🎯 TRANSCRIPTION FINALE OPTIMISÉE GROS16")
    print("Paramètres VAD optimaux de petit16 appliqués")
    print("="*60)
    
    start_time = time.time()
    
    # Paramètres VAD optimaux trouvés pour petit16
    optimal_vad_params = {
        'min_silence_duration_ms': 212,
        'speech_pad_ms': 150,
        'threshold': 0.4
    }
    
    # Configuration Whisper optimale
    config = {
        'language': 'fr',
        'vad_filter': True,
        'word_timestamps': True,
        'temperature': 0.0,
        'condition_on_previous_text': True,
        'beam_size': 3,
        'best_of': 2,
        'patience': 1.5
    }
    
    print(f"🎯 Paramètres VAD utilisés: {optimal_vad_params}")
    print(f"📦 Traitement par chunks de 25s")
    print("-" * 60)
    
    # Analyse du fichier
    characteristics = analyze_audio_characteristics(audio_file)
    total_duration = characteristics['duration']
    chunk_size = 25
    num_chunks = int(np.ceil(total_duration / chunk_size))
    
    print(f"⏱️ Durée totale: {total_duration:.1f}s")
    print(f"📦 Nombre de chunks: {num_chunks}")
    
    all_segments = []
    quality_history = []
    
    for i in range(num_chunks):
        start_chunk = i * chunk_size
        end_chunk = min((i + 1) * chunk_size, total_duration)
        
        print(f"\n📦 Chunk {i+1}/{num_chunks} ({start_chunk:.1f}s - {end_chunk:.1f}s)")
        
        try:
            # Transcription du chunk
            chunk_segments = transcribe_chunk_with_timing(
                audio_file, start_chunk, end_chunk, optimal_vad_params, config
            )
            
            if chunk_segments:
                # Ajustement des timestamps pour le fichier complet
                for segment in chunk_segments:
                    segment.start += start_chunk
                    segment.end += start_chunk
                
                all_segments.extend(chunk_segments)
                
                # Analyse de la qualité du chunk
                chunk_confidence = np.mean([s.avg_logprob for s in chunk_segments])
                quality_history.append(chunk_confidence)
                
                print(f"  ✅ {len(chunk_segments)} segments, confiance: {chunk_confidence:.3f}")
                
                # AFFICHAGE DE LA TRANSCRIPTION DU CHUNK
                print(f"  📝 TRANSCRIPTION CHUNK {i+1}:")
                for j, segment in enumerate(chunk_segments):
                    timestamp_start = format_time(segment.start)
                    timestamp_end = format_time(segment.end)
                    print(f"    {j+1:2d}. [{timestamp_start} - {timestamp_end}] {segment.text}")
                print(f"  " + "-" * 50)
            
            else:
                print(f"  ❌ Aucun segment détecté")
                quality_history.append(-1.0)
        
        except Exception as e:
            print(f"  ❌ Erreur chunk {i+1}: {e}")
            quality_history.append(-1.0)
    
    total_time = time.time() - start_time
    
    if not all_segments:
        print("❌ Aucun segment transcrit")
        return None
    
    # Analyse des résultats
    print(f"\n📊 ANALYSE DES RÉSULTATS FINAUX")
    print("-" * 40)
    
    num_segments = len(all_segments)
    avg_confidence = np.mean([s.avg_logprob for s in all_segments])
    avg_segment_duration = np.mean([s.end - s.start for s in all_segments])
    
    # Analyse du début (premiers 40s) pour comparaison avec petit16
    first_40s_segments = [s for s in all_segments if s.start < 40]
    
    print(f"📝 Segments totaux: {num_segments}")
    print(f"🎭 Confiance moyenne: {avg_confidence:.3f}")
    print(f"⏱️ Durée moyenne/segment: {avg_segment_duration:.1f}s")
    print(f"🕐 Segments premiers 40s: {len(first_40s_segments)}")
    
    if first_40s_segments:
        first_40s_confidence = np.mean([s.avg_logprob for s in first_40s_segments])
        first_segment_start = first_40s_segments[0].start
        print(f"🎯 Confiance début: {first_40s_confidence:.3f}")
        print(f"▶️ Premier segment: {first_segment_start:.1f}s")
    
    # Sauvegarde des résultats
    print(f"\n💾 SAUVEGARDE DES RÉSULTATS")
    print("-" * 40)
    
    output_prefix = "gros16_enhanced_final"
    
    # Transcription principale
    txt_file = f"{output_prefix}_transcription.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"# TRANSCRIPTION FINALE OPTIMISÉE - {audio_file}\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Segments: {num_segments}\n")
        f.write(f"# Confiance moyenne: {avg_confidence:.3f}\n")
        f.write(f"# Temps total: {total_time/60:.1f} min\n")
        f.write(f"# Paramètres VAD: {optimal_vad_params}\n")
        f.write("="*60 + "\n\n")
        
        for segment in all_segments:
            timestamp_start = format_time(segment.start)
            timestamp_end = format_time(segment.end)
            f.write(f"[{timestamp_start} - {timestamp_end}] {segment.text}\n")
    
    # Fichier SRT pour sous-titres
    srt_file = f"{output_prefix}_subtitles.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(all_segments, 1):
            start_time_srt = format_srt_time(segment.start)
            end_time_srt = format_srt_time(segment.end)
            f.write(f"{i}\n")
            f.write(f"{start_time_srt} --> {end_time_srt}\n")
            f.write(f"{segment.text}\n\n")
    
    # Rapport de comparaison final
    comparison_file = f"{output_prefix}_comparison_report.txt"
    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write(f"RAPPORT FINAL - COMPARAISON TOUTES MÉTHODES\n")
        f.write(f"==========================================\n\n")
        
        f.write(f"MÉTHODE FINALE OPTIMISÉE:\n")
        f.write(f"- Segments totaux: {num_segments}\n")
        f.write(f"- Confiance: {avg_confidence:.3f}\n")
        f.write(f"- Segments 40s: {len(first_40s_segments)}\n")
        f.write(f"- Temps total: {total_time/60:.1f} min\n")
        if first_40s_segments:
            f.write(f"- Premier segment: {first_40s_segments[0].start:.1f}s\n")
        
        f.write(f"\nCOMPARAISON AVEC TOUTES LES MÉTHODES:\n")
        f.write(f"1. Progressive: 147 segments, -0.325 conf, premier à 35s, 10.6 min\n")
        f.write(f"2. Deux phases: 5 segments, conf ?, premier à ?s, ? min\n")
        f.write(f"3. Finale optimisée: {num_segments} segments, {avg_confidence:.3f} conf")
        if first_40s_segments:
            f.write(f", premier à {first_40s_segments[0].start:.1f}s")
        f.write(f", {total_time/60:.1f} min\n")
        
        f.write(f"\nRÉFÉRENCE PETIT16 (40s):\n")
        f.write(f"- 6 segments, confiance -0.237, premier à 11s\n")
        
        f.write(f"\nAMÉLIORATIONS vs PROGRESSIVE:\n")
        f.write(f"- Segments: {num_segments - 147:+d}\n")
        f.write(f"- Confiance: {avg_confidence - (-0.325):+.3f}\n")
        if first_40s_segments:
            f.write(f"- Capture début: {35 - first_40s_segments[0].start:+.1f}s plus tôt\n")
        f.write(f"- Temps: {10.6 - total_time/60:+.1f} min plus rapide\n")
    
    print(f"✅ Fichiers sauvegardés:")
    print(f"   📝 Transcription: {txt_file}")
    print(f"   🎬 Sous-titres: {srt_file}")
    print(f"   📊 Rapport: {comparison_file}")
    
    # Statistiques finales de comparaison
    print(f"\n📈 COMPARAISON FINALE AVEC TOUTES LES MÉTHODES")
    print("-" * 50)
    
    print(f"📊 RÉFÉRENCE PETIT16 (40s):")
    print(f"   - 6 segments, confiance -0.237")
    print(f"   - Premier segment: 11.0s")
    
    print(f"📊 MÉTHODE PROGRESSIVE:")
    print(f"   - 147 segments, confiance -0.325")
    print(f"   - Premier segment: 35.0s")
    print(f"   - Temps: 10.6 min")
    
    print(f"📊 MÉTHODE FINALE OPTIMISÉE:")
    print(f"   - {num_segments} segments, confiance {avg_confidence:.3f}")
    if first_40s_segments:
        print(f"   - Premier segment: {first_40s_segments[0].start:.1f}s")
    print(f"   - Temps: {total_time/60:.1f} min")
    
    # Calcul des améliorations
    if first_40s_segments:
        improvement_segments = num_segments - 147
        improvement_confidence = avg_confidence - (-0.325)
        capture_improvement = 35.0 - first_40s_segments[0].start
        time_improvement = 10.6 - total_time/60
        
        print(f"📈 AMÉLIORATIONS vs PROGRESSIVE:")
        print(f"   - Segments: {improvement_segments:+d}")
        print(f"   - Confiance: {improvement_confidence:+.3f}")
        print(f"   - Capture début: {capture_improvement:+.1f}s plus tôt")
        print(f"   - Efficacité: {time_improvement:+.1f} min plus rapide")
        
        # Comparaison avec petit16 (40s)
        petit16_vs_gros16_40s = len(first_40s_segments) - 6
        print(f"📈 vs PETIT16 (40s):")
        print(f"   - Segments 40s: {petit16_vs_gros16_40s:+d} (attendu: ~+3 à +5)")
        if first_40s_segments[0].start <= 11:
            print(f"   - ✅ Capture début équivalente ou meilleure")
        else:
            print(f"   - ⚠️ Capture début: {first_40s_segments[0].start - 11:+.1f}s plus tard")
    
    return {
        'segments': all_segments,
        'vad_params': optimal_vad_params,
        'quality_history': quality_history,
        'stats': {
            'total_time': total_time,
            'confidence': avg_confidence,
            'num_segments': num_segments,
            'first_40s_segments': len(first_40s_segments) if first_40s_segments else 0
        }
    }

def format_srt_time(seconds):
    """Format timestamp pour fichier SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

if __name__ == "__main__":
    print("🎯 TRANSCRIPTION FINALE OPTIMISÉE - GROS16")
    print("Utilise les paramètres VAD optimaux de petit16")
    print("Traitement par chunks de 25s avec affichage détaillé")
    print("="*60)
    
    result = transcribe_gros16_final()
    
    if result:
        print(f"\n🎉 TRANSCRIPTION FINALE RÉUSSIE!")
        print(f"🚀 Méthode optimisée validée")
        print(f"📊 Résultats sauvegardés avec comparaisons complètes")
    else:
        print(f"\n❌ ÉCHEC DE LA TRANSCRIPTION") 