#!/usr/bin/env python3
"""
Script d'analyse audio avancée pour comprendre la diversité vocale.
Analyse les caractéristiques qui permettent de distinguer les locuteurs.
Respecte les règles .cursorrules : ZÉRO valeur codée en dur.
"""

import os
import sys
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Any, Tuple, Dict, List
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configuration du logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def extract_vocal_features(audio: np.ndarray, sr: int, segment_duration: float = 2.0) -> Dict[str, Any]:
    """
    Extraction de caractéristiques vocales détaillées par segments.
    Respecte les règles .cursorrules : paramètres calculés adaptatifs.
    """
    # Calcul de la taille de segment adaptative
    segment_samples = int(segment_duration * sr)
    hop_length = sr // 100  # 10ms hop adaptatif
    
    # Division en segments adaptatifs
    num_segments = len(audio) // segment_samples
    if num_segments < 3:  # Minimum de segments pour analyse
        segment_samples = len(audio) // 3
        num_segments = 3
    
    features = {
        'mfcc_segments': [],
        'pitch_segments': [],
        'spectral_centroid_segments': [],
        'spectral_rolloff_segments': [],
        'zero_crossing_rate_segments': [],
        'chroma_segments': [],
        'spectral_contrast_segments': [],
        'energy_segments': [],
        'formants_segments': []
    }
    
    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = min((i + 1) * segment_samples, len(audio))
        segment = audio[start_idx:end_idx]
        
        if len(segment) < sr // 10:  # Segment trop court
            continue
        
        # MFCC (caractéristiques spectrales)
        n_mfcc = min(13, len(segment) // (sr // 100))  # Adaptatif selon la longueur
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        features['mfcc_segments'].append(np.mean(mfcc, axis=1))
        
        # Pitch (fréquence fondamentale)
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr, hop_length=hop_length)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Pitch valide
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_segments'].append({
                'mean': np.mean(pitch_values),
                'std': np.std(pitch_values),
                'min': np.min(pitch_values),
                'max': np.max(pitch_values)
            })
        else:
            features['pitch_segments'].append({
                'mean': 0, 'std': 0, 'min': 0, 'max': 0
            })
        
        # Centroïde spectral (brillance)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=hop_length)
        features['spectral_centroid_segments'].append(np.mean(spectral_centroid))
        
        # Rolloff spectral
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=hop_length)
        features['spectral_rolloff_segments'].append(np.mean(spectral_rolloff))
        
        # Taux de passage par zéro
        zcr = librosa.feature.zero_crossing_rate(segment, hop_length=hop_length)
        features['zero_crossing_rate_segments'].append(np.mean(zcr))
        
        # Chroma (contenu harmonique)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr, hop_length=hop_length)
        features['chroma_segments'].append(np.mean(chroma, axis=1))
        
        # Contraste spectral
        spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, hop_length=hop_length)
        features['spectral_contrast_segments'].append(np.mean(spectral_contrast, axis=1))
        
        # Énergie
        energy = np.sum(segment ** 2) / len(segment)
        features['energy_segments'].append(energy)
        
        # Estimation des formants (approximation)
        # Utilisation du spectre pour estimer les pics de formants
        fft = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1/sr)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Recherche des pics adaptatifs
        peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        formant_freqs = freqs[peaks][:5]  # 5 premiers formants max
        
        if len(formant_freqs) > 0:
            features['formants_segments'].append({
                'f1': formant_freqs[0] if len(formant_freqs) > 0 else 0,
                'f2': formant_freqs[1] if len(formant_freqs) > 1 else 0,
                'f3': formant_freqs[2] if len(formant_freqs) > 2 else 0
            })
        else:
            features['formants_segments'].append({'f1': 0, 'f2': 0, 'f3': 0})
    
    return features

def analyze_speaker_diversity(features: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Analyse de la diversité des locuteurs basée sur les caractéristiques.
    Respecte les règles .cursorrules : clustering adaptatif.
    """
    if logger:
        logger.info("🔍 Analyse de la diversité des locuteurs...")
    
    # Préparation des données pour clustering
    feature_matrix = []
    
    for i in range(len(features['mfcc_segments'])):
        row = []
        
        # MFCC moyens
        row.extend(features['mfcc_segments'][i])
        
        # Caractéristiques de pitch
        pitch_data = features['pitch_segments'][i]
        row.extend([pitch_data['mean'], pitch_data['std'], pitch_data['min'], pitch_data['max']])
        
        # Autres caractéristiques
        row.append(features['spectral_centroid_segments'][i])
        row.append(features['spectral_rolloff_segments'][i])
        row.append(features['zero_crossing_rate_segments'][i])
        row.append(features['energy_segments'][i])
        
        # Chroma moyens
        row.extend(features['chroma_segments'][i])
        
        # Contraste spectral moyens
        row.extend(features['spectral_contrast_segments'][i])
        
        # Formants
        formant_data = features['formants_segments'][i]
        row.extend([formant_data['f1'], formant_data['f2'], formant_data['f3']])
        
        feature_matrix.append(row)
    
    feature_matrix = np.array(feature_matrix)
    
    # Normalisation adaptative
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Test de clustering avec différents nombres de clusters
    max_clusters = min(10, len(feature_matrix) // 2)  # Adaptatif selon le nombre de segments
    cluster_scores = {}
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Score de silhouette adaptatif
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            score = silhouette_score(feature_matrix_scaled, labels)
            cluster_scores[n_clusters] = score
        else:
            cluster_scores[n_clusters] = -1
    
    # Sélection du nombre optimal de clusters
    best_n_clusters = max(cluster_scores.keys(), key=lambda k: cluster_scores[k])
    best_score = cluster_scores[best_n_clusters]
    
    # Clustering final
    kmeans_final = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    final_labels = kmeans_final.fit_predict(feature_matrix_scaled)
    
    # Analyse des caractéristiques par cluster
    cluster_analysis = {}
    for cluster_id in range(best_n_clusters):
        cluster_mask = final_labels == cluster_id
        cluster_features = feature_matrix[cluster_mask]
        
        cluster_analysis[cluster_id] = {
            'num_segments': np.sum(cluster_mask),
            'avg_pitch': np.mean([features['pitch_segments'][i]['mean'] 
                                for i in range(len(features['pitch_segments'])) 
                                if cluster_mask[i] and features['pitch_segments'][i]['mean'] > 0]),
            'avg_spectral_centroid': np.mean([features['spectral_centroid_segments'][i] 
                                            for i in range(len(features['spectral_centroid_segments'])) 
                                            if cluster_mask[i]]),
            'avg_energy': np.mean([features['energy_segments'][i] 
                                 for i in range(len(features['energy_segments'])) 
                                 if cluster_mask[i]])
        }
    
    if logger:
        logger.info(f"✅ Analyse terminée : {best_n_clusters} groupes vocaux détectés")
        logger.info(f"   Score de silhouette : {best_score:.3f}")
        for cluster_id, analysis in cluster_analysis.items():
            logger.info(f"   Groupe {cluster_id}: {analysis['num_segments']} segments, "
                       f"pitch moyen: {analysis['avg_pitch']:.1f}Hz")
    
    return {
        'optimal_clusters': best_n_clusters,
        'silhouette_score': best_score,
        'cluster_labels': final_labels,
        'cluster_analysis': cluster_analysis,
        'all_scores': cluster_scores,
        'feature_matrix': feature_matrix_scaled
    }

def compare_with_reference(reference_file: str, detected_clusters: int, logger=None) -> Dict[str, Any]:
    """
    Compare les résultats avec le fichier de référence.
    """
    if not os.path.exists(reference_file):
        if logger:
            logger.warning(f"Fichier de référence non trouvé : {reference_file}")
        return {}
    
    # Extraction du nombre de locuteurs de référence
    with open(reference_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Recherche des locuteurs uniques
    import re
    speakers = set(re.findall(r'\[SPEAKER_[^\]]+\]', content))
    reference_speakers = len(speakers)
    
    if logger:
        logger.info(f"📊 Comparaison avec référence :")
        logger.info(f"   Locuteurs de référence : {reference_speakers}")
        logger.info(f"   Groupes vocaux détectés : {detected_clusters}")
        logger.info(f"   Écart : {abs(reference_speakers - detected_clusters)}")
    
    return {
        'reference_speakers': reference_speakers,
        'detected_clusters': detected_clusters,
        'accuracy': min(detected_clusters, reference_speakers) / max(detected_clusters, reference_speakers),
        'difference': abs(reference_speakers - detected_clusters)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyse avancée de la diversité vocale")
    parser.add_argument("audio_file", help="Fichier audio à analyser")
    parser.add_argument("--reference", help="Fichier de référence SRT pour comparaison")
    parser.add_argument("--verbose", action="store_true", help="Mode verbose")
    parser.add_argument("--segment-duration", type=float, default=2.0, 
                       help="Durée des segments d'analyse (défaut: 2.0s)")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    if not os.path.exists(args.audio_file):
        logger.error(f"Fichier audio non trouvé : {args.audio_file}")
        sys.exit(1)
    
    logger.info(f"🎵 Analyse de : {args.audio_file}")
    
    # Chargement de l'audio
    try:
        audio, sr = librosa.load(args.audio_file, sr=None)
        logger.info(f"   Durée : {len(audio)/sr:.1f}s, Fréquence : {sr}Hz")
    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}")
        sys.exit(1)
    
    # Extraction des caractéristiques
    logger.info("🔬 Extraction des caractéristiques vocales...")
    features = extract_vocal_features(audio, sr, args.segment_duration)
    logger.info(f"   {len(features['mfcc_segments'])} segments analysés")
    
    # Analyse de la diversité
    diversity_analysis = analyze_speaker_diversity(features, logger)
    
    # Comparaison avec référence si fournie
    if args.reference:
        comparison = compare_with_reference(args.reference, diversity_analysis['optimal_clusters'], logger)
    
    # Sauvegarde des résultats
    output_file = args.audio_file.replace('.wav', '_vocal_analysis.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ANALYSE VOCALE AVANCÉE\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"FICHIER ANALYSÉ: {args.audio_file}\n")
        f.write(f"DURÉE: {len(audio)/sr:.1f}s\n")
        f.write(f"SEGMENTS ANALYSÉS: {len(features['mfcc_segments'])}\n\n")
        
        f.write("RÉSULTATS DU CLUSTERING:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Groupes vocaux optimaux: {diversity_analysis['optimal_clusters']}\n")
        f.write(f"Score de silhouette: {diversity_analysis['silhouette_score']:.3f}\n\n")
        
        f.write("ANALYSE PAR GROUPE:\n")
        f.write("-" * 30 + "\n")
        for cluster_id, analysis in diversity_analysis['cluster_analysis'].items():
            f.write(f"Groupe {cluster_id}:\n")
            f.write(f"  • Segments: {analysis['num_segments']}\n")
            f.write(f"  • Pitch moyen: {analysis['avg_pitch']:.1f}Hz\n")
            f.write(f"  • Centroïde spectral: {analysis['avg_spectral_centroid']:.1f}Hz\n")
            f.write(f"  • Énergie moyenne: {analysis['avg_energy']:.6f}\n\n")
        
        if args.reference and 'comparison' in locals():
            f.write("COMPARAISON AVEC RÉFÉRENCE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Locuteurs de référence: {comparison['reference_speakers']}\n")
            f.write(f"Groupes détectés: {comparison['detected_clusters']}\n")
            f.write(f"Précision: {comparison['accuracy']:.1%}\n")
            f.write(f"Écart: {comparison['difference']}\n")
    
    logger.info(f"💾 Résultats sauvegardés : {output_file}")
    logger.info("✅ Analyse terminée")

if __name__ == "__main__":
    main() 