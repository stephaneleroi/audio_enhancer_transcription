#!/usr/bin/env python3
"""
Script de transcription adaptative dynamique
Système intelligent qui s'ajuste chunk par chunk selon les caractéristiques audio
"""

import sys
import os
import time
import numpy as np
import librosa
import subprocess
from faster_whisper import WhisperModel

# Prompt initial pour améliorer la qualité de transcription
INITIAL_PROMPT = """
Transcription d'une réunion professionnelle avec plusieurs intervenants.
Certains participants parlent à voix basse ou sont éloignés du microphone.
Le vocabulaire utilisé est formel et technique.
"""

def format_time(seconds):
    """Format timestamp pour lecture humaine"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    else:
        return f"{minutes}m{secs:02d}s"

def analyze_audio_global(audio_file):
    """
    1° ÉTAPE: Analyse globale de l'audio pour déduire des paramètres candidats
    """
    print("🔍 ÉTAPE 1: Analyse globale de l'audio")
    print("-" * 50)
    
    # Charger l'audio
    y, sr = librosa.load(audio_file, sr=16000)
    duration = len(y) / sr
    
    # Analyse énergétique
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    mean_energy = np.mean(energy)
    energy_std = np.std(energy)
    dynamic_range = np.max(energy) - np.min(energy)
    
    # Analyse spectrale
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Détection de silences approximative
    silence_threshold = mean_energy * 0.1
    silent_frames = energy < silence_threshold
    
    # Estimation des durées de silence
    silence_durations = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            silence_duration = (i - silence_start) * (512 / sr) * 1000  # en ms
            if silence_duration > 50:  # Ignorer les très courts silences
                silence_durations.append(silence_duration)
            in_silence = False
    
    # Calcul du débit de parole estimé
    speech_frames = np.sum(~silent_frames)
    speech_rate = speech_frames / (len(silent_frames) / (sr / 512))
    
    characteristics = {
        'duration': duration,
        'mean_energy': mean_energy,
        'energy_std': energy_std,
        'dynamic_range': dynamic_range,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'silence_durations': silence_durations,
        'speech_rate': speech_rate
    }
    
    print(f"⏱️ Durée: {duration:.1f}s")
    print(f"🔊 Énergie moyenne: {mean_energy:.4f}")
    print(f"📊 Plage dynamique: {dynamic_range:.4f}")
    print(f"🎵 Centre spectral: {spectral_centroid:.0f} Hz")
    print(f"🗣️ Débit de parole: {speech_rate:.1f}")
    
    if silence_durations:
        avg_silence = np.mean(silence_durations)
        print(f"🤫 Silence moyen: {avg_silence:.0f}ms")
    
    # Déduction des paramètres candidats
    candidate_params = deduce_candidate_parameters(characteristics)
    
    print(f"\n🎯 Paramètres candidats déduits:")
    for i, params in enumerate(candidate_params):
        print(f"  Candidat {i+1}: threshold={params['threshold']}, silence={params['min_silence_duration_ms']}ms")
    
    return characteristics, candidate_params

def deduce_candidate_parameters(characteristics):
    """
    Déduit 5 configurations candidates basées sur l'analyse audio
    """
    mean_energy = characteristics['mean_energy']
    dynamic_range = characteristics['dynamic_range']
    silence_durations = characteristics['silence_durations']
    
    # Estimation du seuil de base selon l'énergie
    if mean_energy > 0.05:
        base_threshold = 0.5  # Signal fort
    elif mean_energy > 0.02:
        base_threshold = 0.4  # Signal moyen
    else:
        base_threshold = 0.3  # Signal faible
    
    # Estimation de la durée de silence selon les pauses naturelles
    if silence_durations:
        avg_silence = np.mean(silence_durations)
        base_silence = max(100, min(500, int(avg_silence * 0.8)))
    else:
        base_silence = 300
    
    # Génération de 5 candidats avec variations
    candidates = [
        # Candidat 1: Configuration de base
        {
            'threshold': base_threshold,
            'min_silence_duration_ms': base_silence,
            'speech_pad_ms': 150
        },
        # Candidat 2: Plus sensible (capture plus)
        {
            'threshold': max(0.2, base_threshold - 0.1),
            'min_silence_duration_ms': max(100, base_silence - 100),
            'speech_pad_ms': 200
        },
        # Candidat 3: Plus restrictif (moins de bruit)
        {
            'threshold': min(0.6, base_threshold + 0.1),
            'min_silence_duration_ms': min(600, base_silence + 150),
            'speech_pad_ms': 100
        },
        # Candidat 4: Optimisé pour parole rapide
        {
            'threshold': base_threshold,
            'min_silence_duration_ms': max(100, int(base_silence * 0.6)),
            'speech_pad_ms': 250
        },
        # Candidat 5: Optimisé pour parole lente/réfléchie
        {
            'threshold': base_threshold,
            'min_silence_duration_ms': min(800, int(base_silence * 1.5)),
            'speech_pad_ms': 100
        }
    ]
    
    return candidates

def extract_calibration_sample(audio_file, duration=15):
    """
    Extrait les 15 premières secondes pour calibration
    """
    sample_file = f"temp_calibration_{int(time.time())}.wav"
    
    cmd = [
        'ffmpeg', '-i', audio_file,
        '-ss', '0',
        '-t', str(duration),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-y', sample_file
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    return sample_file

def test_vad_configuration(audio_file, vad_params, model):
    """
    Teste une configuration VAD et retourne un score de qualité
    """
    try:
        segments, _ = model.transcribe(
            audio_file,
            language='fr',
            vad_filter=True,
            vad_parameters=vad_params,
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=True,
            beam_size=3,
            best_of=2,
            patience=1.5,
            initial_prompt=INITIAL_PROMPT
        )
        
        segments_list = list(segments)
        
        if not segments_list:
            return 0, 0, -2.0
        
        # Calcul du score basé sur nombre de segments et confiance
        num_segments = len(segments_list)
        avg_confidence = np.mean([s.avg_logprob for s in segments_list])
        avg_duration = np.mean([s.end - s.start for s in segments_list])
        
        # Score composite (plus c'est haut, mieux c'est)
        score = num_segments * 0.3 + (-avg_confidence) * 0.5 + (1/max(0.5, avg_duration)) * 0.2
        
        return score, num_segments, avg_confidence
        
    except Exception as e:
        print(f"    ❌ Erreur test: {e}")
        return 0, 0, -2.0

def calibrate_on_sample(audio_file, candidate_params):
    """
    2° ÉTAPE: Calibration sur 15s avec 5 essais pour trouver les meilleurs paramètres
    """
    print(f"\n🧪 ÉTAPE 2: Calibration sur échantillon 15s")
    print("-" * 50)
    
    # Extraire l'échantillon
    sample_file = extract_calibration_sample(audio_file, 15)
    
    try:
        # Charger le modèle Whisper
        model = WhisperModel('large-v3', device='cpu', compute_type='int8')
        
        best_score = -1
        best_params = None
        best_stats = None
        
        print("🔬 Test des 5 configurations candidates:")
        
        for i, params in enumerate(candidate_params):
            print(f"  Test {i+1}: threshold={params['threshold']}, silence={params['min_silence_duration_ms']}ms")
            
            score, num_segments, confidence = test_vad_configuration(sample_file, params, model)
            
            print(f"    📊 Score: {score:.3f}, Segments: {num_segments}, Confiance: {confidence:.3f}")
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_stats = (score, num_segments, confidence)
        
        print(f"\n🏆 MEILLEURE CONFIGURATION:")
        print(f"  Paramètres: {best_params}")
        print(f"  Score: {best_stats[0]:.3f}, Segments: {best_stats[1]}, Confiance: {best_stats[2]:.3f}")
        
        return best_params, model
        
    finally:
        # Nettoyage
        if os.path.exists(sample_file):
            os.remove(sample_file)

def analyze_chunk_characteristics(audio_file, start_time, end_time):
    """
    Analyse les caractéristiques spécifiques d'un chunk
    """
    # Charger seulement le chunk
    y, sr = librosa.load(audio_file, sr=16000, offset=start_time, duration=end_time-start_time)
    
    if len(y) == 0:
        return None
    
    # Analyse énergétique du chunk
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    mean_energy = np.mean(energy)
    energy_std = np.std(energy)
    
    # Détection de variabilité
    energy_variability = energy_std / (mean_energy + 1e-8)
    
    # Analyse spectrale
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    return {
        'mean_energy': mean_energy,
        'energy_variability': energy_variability,
        'spectral_centroid': spectral_centroid,
        'duration': len(y) / sr
    }

def adjust_parameters_for_chunk(base_params, chunk_characteristics, global_characteristics):
    """
    Ajuste les paramètres VAD selon les caractéristiques du chunk
    """
    if chunk_characteristics is None:
        return base_params
    
    adjusted_params = base_params.copy()
    
    # Ajustement selon l'énergie relative
    energy_ratio = chunk_characteristics['mean_energy'] / (global_characteristics['mean_energy'] + 1e-8)
    
    if energy_ratio < 0.7:  # Chunk plus faible que la moyenne
        adjusted_params['threshold'] = max(0.2, adjusted_params['threshold'] - 0.1)
        adjusted_params['speech_pad_ms'] = min(300, adjusted_params['speech_pad_ms'] + 50)
    elif energy_ratio > 1.3:  # Chunk plus fort que la moyenne
        adjusted_params['threshold'] = min(0.6, adjusted_params['threshold'] + 0.1)
        adjusted_params['speech_pad_ms'] = max(100, adjusted_params['speech_pad_ms'] - 50)
    
    # Ajustement selon la variabilité
    if chunk_characteristics['energy_variability'] > 2.0:  # Très variable
        adjusted_params['min_silence_duration_ms'] = max(100, adjusted_params['min_silence_duration_ms'] - 50)
    elif chunk_characteristics['energy_variability'] < 0.5:  # Très stable
        adjusted_params['min_silence_duration_ms'] = min(600, adjusted_params['min_silence_duration_ms'] + 100)
    
    return adjusted_params

def transcribe_chunk_with_timing(audio_file, start_time, end_time, vad_params, model):
    """
    Transcrit un chunk spécifique avec timing précis
    """
    try:
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
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Transcription
        segments, _ = model.transcribe(
            chunk_file,
            language='fr',
            vad_filter=True,
            vad_parameters=vad_params,
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=True,
            beam_size=3,
            best_of=2,
            patience=1.5,
            initial_prompt=INITIAL_PROMPT
        )
        
        return list(segments)
        
    except Exception as e:
        print(f"    ❌ Erreur transcription chunk: {e}")
        return []
    finally:
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

def adaptive_transcription_main():
    """
    3° et 4° ÉTAPES: Traitement adaptatif chunk par chunk
    """
    audio_file = "gros16_enhanced.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Fichier {audio_file} non trouvé")
        return
    
    start_time = time.time()
    
    # ÉTAPE 1: Analyse globale
    global_characteristics, candidate_params = analyze_audio_global(audio_file)
    
    # ÉTAPE 2: Calibration sur échantillon
    best_params, model = calibrate_on_sample(audio_file, candidate_params)
    
    # ÉTAPES 3 & 4: Traitement adaptatif chunk par chunk
    print(f"\n🚀 ÉTAPES 3 & 4: Traitement adaptatif chunk par chunk")
    print("-" * 60)
    
    total_duration = global_characteristics['duration']
    chunk_size = 25  # secondes
    num_chunks = int(np.ceil(total_duration / chunk_size))
    
    print(f"⏱️ Durée totale: {total_duration:.1f}s")
    print(f"📦 Nombre de chunks: {num_chunks}")
    print(f"🎯 Paramètres de base: {best_params}")
    
    all_segments = []
    
    for i in range(num_chunks):
        start_chunk = i * chunk_size
        end_chunk = min((i + 1) * chunk_size, total_duration)
        
        print(f"\n📦 CHUNK {i+1}/{num_chunks} ({start_chunk:.1f}s - {end_chunk:.1f}s)")
        
        # Analyser les caractéristiques du chunk
        chunk_characteristics = analyze_chunk_characteristics(audio_file, start_chunk, end_chunk)
        
        # Ajuster les paramètres pour ce chunk
        adjusted_params = adjust_parameters_for_chunk(best_params, chunk_characteristics, global_characteristics)
        
        # Afficher les ajustements si différents
        if adjusted_params != best_params:
            print(f"  🔧 Paramètres ajustés: threshold={adjusted_params['threshold']:.1f}, silence={adjusted_params['min_silence_duration_ms']}ms")
        else:
            print(f"  ✅ Paramètres de base conservés")
        
        # Transcription du chunk
        chunk_segments = transcribe_chunk_with_timing(
            audio_file, start_chunk, end_chunk, adjusted_params, model
        )
        
        if chunk_segments:
            # Ajustement des timestamps pour le fichier complet
            for segment in chunk_segments:
                segment.start += start_chunk
                segment.end += start_chunk
            
            all_segments.extend(chunk_segments)
            
            # Analyse de la qualité du chunk
            chunk_confidence = np.mean([s.avg_logprob for s in chunk_segments])
            
            print(f"  ✅ {len(chunk_segments)} segments, confiance: {chunk_confidence:.3f}")
            
            # AFFICHAGE DE LA TRANSCRIPTION DU CHUNK
            print(f"  📝 TRANSCRIPTION:")
            for j, segment in enumerate(chunk_segments):
                timestamp_start = format_time(segment.start)
                timestamp_end = format_time(segment.end)
                print(f"    {j+1:2d}. [{timestamp_start} - {timestamp_end}] {segment.text}")
        else:
            print(f"  ❌ Aucun segment détecté")
        
        print(f"  " + "-" * 50)
    
    # Résultats finaux
    total_time = time.time() - start_time
    
    if all_segments:
        print(f"\n🎉 TRANSCRIPTION TERMINÉE")
        print("-" * 40)
        print(f"📝 Segments totaux: {len(all_segments)}")
        print(f"🎭 Confiance moyenne: {np.mean([s.avg_logprob for s in all_segments]):.3f}")
        print(f"⏱️ Temps total: {format_time(total_time)}")
        
        # Sauvegarde
        save_results(all_segments, audio_file)
    else:
        print("❌ Aucun segment transcrit")

def save_results(segments, audio_file):
    """Sauvegarde les résultats"""
    base_name = os.path.splitext(audio_file)[0]
    
    # Fichier texte
    with open(f"{base_name}_adaptive_transcription.txt", 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            f.write(f"{i+1:3d}. [{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}\n")
    
    # Fichier SRT
    with open(f"{base_name}_adaptive_subtitles.srt", 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            start_time = f"{int(segment.start//3600):02d}:{int((segment.start%3600)//60):02d}:{segment.start%60:06.3f}".replace('.', ',')
            end_time = f"{int(segment.end//3600):02d}:{int((segment.end%3600)//60):02d}:{segment.end%60:06.3f}".replace('.', ',')
            
            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment.text}\n\n")
    
    print(f"💾 Fichiers sauvegardés: {base_name}_adaptive_transcription.txt et {base_name}_adaptive_subtitles.srt")

if __name__ == "__main__":
    adaptive_transcription_main() 