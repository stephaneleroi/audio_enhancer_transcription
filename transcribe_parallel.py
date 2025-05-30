#!/usr/bin/env python3
"""
Script de transcription adaptative multi-threadée
Optimisé pour MacBook M3 avec 16 cœurs et 128GB RAM
Système intelligent qui s'ajuste chunk par chunk avec traitement parallèle

Usage: python transcribe_parallel.py [fichier_audio.wav]
"""

import sys
import os
import time
import numpy as np
import librosa
import subprocess
from faster_whisper import WhisperModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import threading
from queue import Queue, PriorityQueue
import psutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pickle
import argparse

# Prompt initial pour améliorer la qualité de transcription
# DÉSACTIVÉ: Cause des hallucinations avec le mot "réunion"
INITIAL_PROMPT = ""

@dataclass
class ChunkTask:
    """Représente une tâche de transcription d'un chunk"""
    index: int
    start_time: float
    end_time: float
    audio_file: str
    vad_params: Dict[str, Any]
    chunk_characteristics: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other):
        return self.index < other.index

@dataclass
class ChunkResult:
    """Résultat de transcription d'un chunk"""
    index: int
    segments: List[Any]
    confidence: float
    processing_time: float
    worker_id: int
    success: bool = True
    error: Optional[str] = None

def get_optimal_worker_count():
    """
    Détermine le nombre optimal de workers selon les ressources système
    """
    cpu_cores = cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Estimation: chaque modèle Whisper large-v3 utilise ~4GB RAM
    max_workers_by_ram = int(ram_gb / 6)  # Marge de sécurité
    max_workers_by_cpu = min(cpu_cores - 2, 8)  # Garde 2 cœurs pour le système
    
    optimal_workers = min(max_workers_by_ram, max_workers_by_cpu, 6)
    
    print(f"🖥️ Ressources système détectées:")
    print(f"   CPU: {cpu_cores} cœurs")
    print(f"   RAM: {ram_gb:.1f} GB")
    print(f"   Workers optimaux: {optimal_workers}")
    
    return optimal_workers

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
    
    # Extraire échantillon de 15s
    sample_file = extract_calibration_sample(audio_file, 15)
    
    try:
        # Charger le modèle pour calibration
        model = WhisperModel('large-v3', device='cpu', compute_type='int8')
        
        best_score = 0
        best_params = None
        results = []
        
        print(f"🔬 Test de {len(candidate_params)} configurations candidates:")
        
        for i, params in enumerate(candidate_params):
            print(f"  Candidat {i+1}: threshold={params['threshold']:.2f}, silence={params['min_silence_duration_ms']}ms", end=" ")
            
            score, num_segments, confidence = test_vad_configuration(sample_file, params, model)
            results.append((score, num_segments, confidence, params))
            
            print(f"→ Score: {score:.2f}, Segments: {num_segments}, Confiance: {confidence:.3f}")
            
            if score > best_score:
                best_score = score
                best_params = params
        
        # Afficher le classement
        results.sort(reverse=True, key=lambda x: x[0])
        print(f"\n📊 Classement par score:")
        for i, (score, segments, conf, params) in enumerate(results):
            marker = "🏆" if i == 0 else f"  {i+1}."
            print(f"{marker} Score {score:.2f}: threshold={params['threshold']:.2f}, "
                  f"silence={params['min_silence_duration_ms']}ms "
                  f"({segments} segments, conf: {conf:.3f})")
        
        print(f"\n🎯 PARAMÈTRES SÉLECTIONNÉS:")
        print(f"  Meilleurs paramètres: {best_params}")
        print(f"  Score: {best_score:.2f}")
        
        return best_params
        
    finally:
        # Nettoyage
        if os.path.exists(sample_file):
            os.remove(sample_file)

def analyze_chunk_characteristics(audio_file, start_time, end_time):
    """
    Analyse les caractéristiques spécifiques d'un chunk
    """
    try:
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
    except Exception as e:
        print(f"⚠️ Erreur analyse chunk {start_time}-{end_time}: {e}")
        return None

def adjust_parameters_for_chunk(base_params, chunk_characteristics, global_characteristics):
    """
    Ajuste très conservativement les paramètres VAD selon les caractéristiques du chunk
    Priorité à la préservation des segments existants
    """
    if chunk_characteristics is None:
        return base_params
    
    adjusted_params = base_params.copy()
    
    # Ajustement selon l'énergie relative - TRÈS CONSERVATEUR
    energy_ratio = chunk_characteristics['mean_energy'] / (global_characteristics['mean_energy'] + 1e-8)
    
    # Seuils plus stricts pour éviter les sur-ajustements
    if energy_ratio < 0.5:  # Seulement les chunks VRAIMENT très faibles
        adjusted_params['threshold'] = max(0.32, adjusted_params['threshold'] - 0.05)
        adjusted_params['speech_pad_ms'] = min(200, adjusted_params['speech_pad_ms'] + 25)
    elif energy_ratio > 2.0:  # Seulement les chunks VRAIMENT très forts
        adjusted_params['threshold'] = min(0.5, adjusted_params['threshold'] + 0.05)
        adjusted_params['speech_pad_ms'] = max(120, adjusted_params['speech_pad_ms'] - 20)
    
    # Ajustement selon la variabilité énergétique - MINIMAL
    if chunk_characteristics['energy_variability'] > 3.0:  # Seulement extrême variabilité
        adjusted_params['min_silence_duration_ms'] = max(180, adjusted_params['min_silence_duration_ms'] - 20)
    elif chunk_characteristics['energy_variability'] < 0.3:  # Seulement extrême stabilité
        adjusted_params['min_silence_duration_ms'] = min(280, adjusted_params['min_silence_duration_ms'] + 30)
    
    # Ajustement spectral - ADAPTATIF pour petits fichiers
    spectral_center = chunk_characteristics.get('spectral_centroid', 1600)
    if spectral_center < 1200:  # Audio de qualité faible (typique petits fichiers)
        adjusted_params['threshold'] = max(0.20, adjusted_params['threshold'] - 0.15)  # Beaucoup plus sensible
        adjusted_params['speech_pad_ms'] = min(300, adjusted_params['speech_pad_ms'] + 100)
        adjusted_params['min_silence_duration_ms'] = max(100, adjusted_params['min_silence_duration_ms'] - 80)
    elif spectral_center > 2500:  # Audio de très haute qualité
        adjusted_params['threshold'] = min(0.5, adjusted_params['threshold'] + 0.03)
    
    return adjusted_params

def transcribe_chunk_worker(task_data):
    """
    Worker function pour transcription parallèle d'un chunk avec adaptation
    """
    try:
        # Désérialiser la tâche
        task = pickle.loads(task_data)
        worker_id = os.getpid()
        start_time = time.time()
        
        # Créer un modèle Whisper pour ce worker
        model = WhisperModel('large-v3', device='cpu', compute_type='int8')
        
        # Extraire le chunk audio
        chunk_file = f"temp_chunk_{worker_id}_{task.index}_{int(time.time())}.wav"
        
        cmd = [
            'ffmpeg', '-i', task.audio_file,
            '-ss', str(task.start_time),
            '-t', str(task.end_time - task.start_time),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-y', chunk_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Transcription avec paramètres adaptés
        segments, _ = model.transcribe(
            chunk_file,
            language='fr',
            vad_filter=True,
            vad_parameters=task.vad_params,
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=True,
            beam_size=3,
            best_of=2,
            patience=1.5,
            initial_prompt=INITIAL_PROMPT
        )
        
        segments_list = list(segments)
        
        # Ajuster les timestamps pour le fichier complet
        for segment in segments_list:
            segment.start += task.start_time
            segment.end += task.start_time
        
        # Calculer la confiance
        confidence = np.mean([s.avg_logprob for s in segments_list]) if segments_list else -2.0
        
        processing_time = time.time() - start_time
        
        # Informations sur l'adaptation utilisée
        adaptation_info = f"threshold={task.vad_params['threshold']:.2f}, silence={task.vad_params['min_silence_duration_ms']}ms"
        
        # Nettoyage
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
        
        return ChunkResult(
            index=task.index,
            segments=segments_list,
            confidence=confidence,
            processing_time=processing_time,
            worker_id=worker_id,
            success=True,
            error=adaptation_info
        )
        
    except Exception as e:
        # Nettoyage en cas d'erreur
        if 'chunk_file' in locals() and os.path.exists(chunk_file):
            os.remove(chunk_file)
            
        return ChunkResult(
            index=task.index,
            segments=[],
            confidence=-2.0,
            processing_time=time.time() - start_time if 'start_time' in locals() else 0,
            worker_id=os.getpid(),
            success=False,
            error=str(e)
        )

def parallel_transcription_main(audio_file=None):
    """
    3° et 4° ÉTAPES: Traitement adaptatif parallèle chunk par chunk
    """
    if audio_file is None:
        audio_file = "gros16_enhanced.wav"  # Défaut
    
    if not os.path.exists(audio_file):
        print(f"❌ Fichier {audio_file} non trouvé")
        return
    
    print(f"🎵 Fichier audio: {audio_file}")
    start_time = time.time()
    
    # ÉTAPE 1: Analyse globale
    global_characteristics, candidate_params = analyze_audio_global(audio_file)
    
    # ÉTAPE 2: Calibration sur échantillon (utilise les paramètres optimaux)
    best_params = calibrate_on_sample(audio_file, candidate_params)
    
    # ÉTAPE 3: Préparation des chunks avec analyse adaptative
    print(f"\n🔍 ÉTAPE 3: Analyse et préparation des chunks adaptatifs")
    print("-" * 60)
    
    total_duration = global_characteristics['duration']
    chunk_size = 30  # secondes
    num_chunks = int(np.ceil(total_duration / chunk_size))
    optimal_workers = get_optimal_worker_count()
    
    print(f"⏱️ Durée totale: {total_duration:.1f}s")
    print(f"📦 Nombre de chunks: {num_chunks}")
    print(f"🎯 Paramètres de base: {best_params}")
    print(f"🚀 Workers parallèles: {optimal_workers}")
    print(f"🧠 Mode: Adaptation dynamique par chunk")
    
    # Pré-analyser tous les chunks et adapter les paramètres
    tasks = []
    adaptations_count = 0
    print(f"\n📊 Adaptation des paramètres par chunk:")
    
    for i in range(num_chunks):
        start_chunk = i * chunk_size
        end_chunk = min((i + 1) * chunk_size, total_duration)
        
        print(f"  Chunk {i+1}/{num_chunks} ({start_chunk:.1f}s - {end_chunk:.1f}s)", end=" ")
        
        # Analyser les caractéristiques du chunk
        chunk_characteristics = analyze_chunk_characteristics(audio_file, start_chunk, end_chunk)
        
        # Ajuster les paramètres pour ce chunk
        adjusted_params = adjust_parameters_for_chunk(best_params, chunk_characteristics, global_characteristics)
        
        if adjusted_params != best_params:
            adaptations_count += 1
            print(f"🔧 Adapté (th={adjusted_params['threshold']:.2f}, sil={adjusted_params['min_silence_duration_ms']}ms)")
        else:
            print(f"✅ Standard")
        
        # Créer la tâche
        task = ChunkTask(
            index=i,
            start_time=start_chunk,
            end_time=end_chunk,
            audio_file=audio_file,
            vad_params=adjusted_params,
            chunk_characteristics=chunk_characteristics
        )
        
        tasks.append(pickle.dumps(task))
    
    print(f"\n🔧 Adaptations prévues: {adaptations_count}/{num_chunks} chunks ({adaptations_count/num_chunks*100:.1f}%)")
    
    # ÉTAPE 4: Transcription parallèle adaptative
    print(f"\n🚀 ÉTAPE 4: Transcription parallèle adaptative")
    print("-" * 60)
    
    all_segments = []
    results = [None] * num_chunks  # Pour maintenir l'ordre
    completed_chunks = 0
    
    # Lancer le traitement parallèle
    with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        # Soumettre toutes les tâches
        future_to_index = {
            executor.submit(transcribe_chunk_worker, task_data): i 
            for i, task_data in enumerate(tasks)
        }
        
        # Traiter les résultats au fur et à mesure
        for future in as_completed(future_to_index):
            chunk_index = future_to_index[future]
            
            try:
                result = future.result()
                results[result.index] = result
                completed_chunks += 1
                
                if result.success:
                    print(f"✅ Chunk {result.index+1}/{num_chunks} terminé: "
                          f"{len(result.segments)} segments, "
                          f"confiance: {result.confidence:.3f}, "
                          f"temps: {result.processing_time:.1f}s "
                          f"(worker {result.worker_id}) - {result.error}")
                else:
                    print(f"❌ Chunk {result.index+1}/{num_chunks} échoué: {result.error}")
                
                # Afficher le progrès
                progress = (completed_chunks / num_chunks) * 100
                print(f"📊 Progrès global: {progress:.1f}% ({completed_chunks}/{num_chunks})")
                
            except Exception as e:
                print(f"❌ Erreur traitement chunk {chunk_index+1}: {e}")
    
    # Assembler les résultats dans l'ordre
    print(f"\n🔧 Assemblage des résultats...")
    
    for result in results:
        if result and result.success and result.segments:
            all_segments.extend(result.segments)
    
    # Résultats finaux
    total_time = time.time() - start_time
    
    if all_segments:
        print(f"\n🎉 TRANSCRIPTION ADAPTATIVE TERMINÉE")
        print("-" * 50)
        print(f"📝 Segments totaux: {len(all_segments)}")
        print(f"🎭 Confiance moyenne: {np.mean([s.avg_logprob for s in all_segments]):.3f}")
        print(f"⏱️ Temps total: {format_time(total_time)}")
        print(f"🚀 Accélération estimée: ~{optimal_workers:.1f}x vs séquentiel")
        print(f"🧠 Adaptations appliquées: {adaptations_count}/{num_chunks} chunks")
        
        # Statistiques par worker
        worker_stats = {}
        for result in results:
            if result and result.success:
                worker_id = result.worker_id
                if worker_id not in worker_stats:
                    worker_stats[worker_id] = {'chunks': 0, 'time': 0, 'segments': 0}
                worker_stats[worker_id]['chunks'] += 1
                worker_stats[worker_id]['time'] += result.processing_time
                worker_stats[worker_id]['segments'] += len(result.segments)
        
        print(f"\n📊 Statistiques workers:")
        for worker_id, stats in worker_stats.items():
            avg_time = stats['time'] / stats['chunks']
            print(f"  Worker {worker_id}: {stats['chunks']} chunks, "
                  f"{stats['segments']} segments, "
                  f"temps moyen: {avg_time:.1f}s/chunk")
        
        # Sauvegarde
        save_results(all_segments, audio_file, "_adaptive")
        
    else:
        print("❌ Aucun segment transcrit")

def save_results(segments, audio_file, suffix=""):
    """Sauvegarde les résultats"""
    base_name = os.path.splitext(audio_file)[0]
    
    # Fichier texte
    txt_file = f"{base_name}{suffix}_transcription.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            f.write(f"{i+1:3d}. [{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}\n")
    
    # Fichier SRT
    srt_file = f"{base_name}{suffix}_subtitles.srt"
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            start_time = f"{int(segment.start//3600):02d}:{int((segment.start%3600)//60):02d}:{segment.start%60:06.3f}".replace('.', ',')
            end_time = f"{int(segment.end//3600):02d}:{int((segment.end%3600)//60):02d}:{segment.end%60:06.3f}".replace('.', ',')
            
            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment.text}\n\n")
    
    print(f"💾 Fichiers sauvegardés: {txt_file} et {srt_file}")

if __name__ == "__main__":
    print("🚀 TRANSCRIPTION ADAPTATIVE PARALLÈLE")
    print("Optimisée pour MacBook M3 - Multi-threading intelligent")
    print("="*70)
    
    parser = argparse.ArgumentParser(description="Script de transcription adaptative multi-threadée")
    parser.add_argument("audio_file", nargs='?', help="Fichier audio à transcrire")
    args = parser.parse_args()
    
    if args.audio_file:
        audio_file = args.audio_file
    else:
        audio_file = "gros16_enhanced.wav"  # Défaut si aucun fichier fourni
    
    parallel_transcription_main(audio_file) 