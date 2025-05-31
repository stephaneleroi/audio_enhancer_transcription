#!/usr/bin/env python3
"""
Script de test Verbatim via CLI et comparaison avec notre système adaptatif
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('verbatim_cli_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_verbatim_cli():
    """
    Lance Verbatim via CLI avec tous les paramètres pour 5 locuteurs
    """
    logger.info("=== DÉMARRAGE TEST VERBATIM CLI ===")
    
    audio_file = "gros_adaptive_enhanced.mp3"
    output_dir = "verbatim_output"
    
    # Vérification fichier audio
    if not os.path.exists(audio_file):
        logger.error(f"Fichier audio non trouvé: {audio_file}")
        return None
    
    file_size = os.path.getsize(audio_file) / (1024 * 1024)
    logger.info(f"Fichier audio: {audio_file} ({file_size:.1f} MB)")
    
    # Nettoyage et création du dossier de sortie
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Commande Verbatim complète
    cmd = [
        "verbatim", 
        audio_file,
        "-o", output_dir,
        "--languages", "fr", "en",
        "--diarize", "5",  # 5 locuteurs
        "--separate",      # Séparation des voix
        "--docx",          # Format Word
        "--json",          # Format JSON pour analyse
        "--txt",           # Format texte
        "-vv"              # Verbosité maximale
    ]
    
    logger.info(f"Commande: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Exécution de Verbatim
        logger.info("Démarrage de Verbatim...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes maximum
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Code de retour: {result.returncode}")
        logger.info(f"Temps de traitement: {processing_time:.1f}s ({processing_time/60:.1f}min)")
        
        if result.returncode == 0:
            logger.info("Verbatim terminé avec succès")
            if result.stdout:
                logger.info("Sortie standard:")
                logger.info(result.stdout)
        else:
            logger.error(f"Erreur Verbatim (code {result.returncode})")
            if result.stderr:
                logger.error("Erreur stderr:")
                logger.error(result.stderr)
        
        return {
            'success': result.returncode == 0,
            'processing_time': processing_time,
            'output_dir': output_dir,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        logger.error("Timeout: Verbatim a pris plus de 30 minutes")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        return None

def analyze_verbatim_output(verbatim_result):
    """
    Analyse les fichiers de sortie de Verbatim
    """
    if not verbatim_result or not verbatim_result['success']:
        logger.error("Pas de résultats Verbatim à analyser")
        return None
    
    output_dir = verbatim_result['output_dir']
    logger.info("=== ANALYSE FICHIERS VERBATIM ===")
    
    # Liste des fichiers générés
    output_files = list(Path(output_dir).rglob("*"))
    logger.info(f"Fichiers générés: {len([f for f in output_files if f.is_file()])}")
    
    for file_path in output_files:
        if file_path.is_file():
            size_kb = file_path.stat().st_size / 1024
            logger.info(f"  - {file_path.name}: {size_kb:.1f} KB")
    
    # Recherche et analyse du fichier JSON principal
    json_files = list(Path(output_dir).rglob("*.json"))
    logger.info(f"Fichiers JSON trouvés: {len(json_files)}")
    
    for json_file in json_files:
        logger.info(f"Fichier JSON: {json_file}")
    
    if not json_files:
        logger.warning("Aucun fichier JSON trouvé pour l'analyse")
        return None
    
    # Analyse du premier fichier JSON trouvé
    main_json = json_files[0]
    logger.info(f"Analyse du fichier: {main_json}")
    
    try:
        with open(main_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info("Structure JSON:")
        logger.info(f"  - Clés principales: {list(data.keys())}")
        
        # Extraction des métriques selon la structure Verbatim
        segments = data.get('segments', [])
        if not segments and 'utterances' in data:
            segments = data['utterances']
        if not segments and 'words' in data:
            # Si on a que des mots, on les compte
            segments = [data]
        
        total_segments = len(segments)
        
        # Calcul de la confiance
        confidences = []
        speakers = set()
        total_duration = 0
        
        for item in segments:
            # Confiance
            if 'confidence' in item:
                confidences.append(item['confidence'])
            elif 'probability' in item:
                confidences.append(item['probability'])
            elif 'words' in item:
                word_confidences = [word.get('confidence', word.get('probability', 0)) 
                                  for word in item['words']]
                confidences.extend([c for c in word_confidences if c > 0])
            
            # Locuteurs
            speaker = item.get('speaker', item.get('label', item.get('speaker_id', 'Unknown')))
            if speaker and speaker != 'Unknown':
                speakers.add(speaker)
            
            # Durée
            end_time = item.get('end', item.get('stop', item.get('timestamp_end', 0)))
            if end_time > total_duration:
                total_duration = end_time
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        metrics = {
            'total_segments': total_segments,
            'avg_confidence': avg_confidence,
            'speakers_detected': len(speakers),
            'speakers_list': sorted(list(speakers)),
            'total_duration': total_duration,
            'processing_time': verbatim_result['processing_time'],
            'confidence_count': len(confidences)
        }
        
        logger.info("=== MÉTRIQUES VERBATIM ===")
        logger.info(f"Segments totaux: {total_segments}")
        logger.info(f"Confiance moyenne: {avg_confidence:.3f} (sur {len(confidences)} valeurs)")
        logger.info(f"Locuteurs détectés: {len(speakers)}")
        logger.info(f"Liste des locuteurs: {sorted(list(speakers))}")
        logger.info(f"Durée totale: {total_duration:.1f}s")
        logger.info(f"Temps de traitement: {verbatim_result['processing_time']:.1f}s")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse JSON: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def compare_systems(verbatim_metrics):
    """
    Compare Verbatim avec notre système adaptatif
    """
    logger.info("=== COMPARAISON FINALE ===")
    
    # Métriques de notre système (résultats historiques)
    our_system = {
        'name': 'Notre système adaptatif (transcribe_parallel.py)',
        'total_segments': 355,
        'avg_confidence': -0.227,  # Log probabilité
        'processing_time': 8*60 + 9,  # 8min09s
        'speakers_expected': 5,
        'adaptations': '4/32 chunks (12.5%)',
        'technology': 'Faster Whisper Large-V3 + VAD adaptatif'
    }
    
    verbatim_system = {
        'name': 'Verbatim',
        'technology': 'Faster Whisper Large-V3 + PyAnnote + Demucs'
    }
    
    if verbatim_metrics:
        verbatim_system.update(verbatim_metrics)
    
    logger.info("NOTRE SYSTÈME:")
    logger.info(f"  - Technologie: {our_system['technology']}")
    logger.info(f"  - Segments: {our_system['total_segments']}")
    logger.info(f"  - Confiance: {our_system['avg_confidence']} (log)")
    logger.info(f"  - Temps: {our_system['processing_time']}s ({our_system['processing_time']/60:.1f}min)")
    logger.info(f"  - Adaptations: {our_system['adaptations']}")
    
    logger.info("\nVERBATIM:")
    logger.info(f"  - Technologie: {verbatim_system['technology']}")
    
    if verbatim_metrics:
        logger.info(f"  - Segments: {verbatim_metrics['total_segments']}")
        logger.info(f"  - Confiance: {verbatim_metrics['avg_confidence']:.3f} (probabilité)")
        logger.info(f"  - Temps: {verbatim_metrics['processing_time']:.1f}s ({verbatim_metrics['processing_time']/60:.1f}min)")
        logger.info(f"  - Locuteurs: {verbatim_metrics['speakers_detected']}/{our_system['speakers_expected']}")
        logger.info(f"  - IDs locuteurs: {verbatim_metrics['speakers_list']}")
        
        # Comparaisons numériques
        segment_diff = verbatim_metrics['total_segments'] - our_system['total_segments']
        segment_pct = (segment_diff / our_system['total_segments']) * 100
        
        time_diff = verbatim_metrics['processing_time'] - our_system['processing_time']
        time_pct = (time_diff / our_system['processing_time']) * 100
        
        logger.info("\nCOMPARAISON:")
        logger.info(f"  - Différence segments: {segment_diff:+d} ({segment_pct:+.1f}%)")
        logger.info(f"  - Différence temps: {time_diff:+.1f}s ({time_pct:+.1f}%)")
        
        # Évaluation de la diarisation
        if verbatim_metrics['speakers_detected'] == our_system['speakers_expected']:
            logger.info(f"  - Diarisation: ✓ {verbatim_metrics['speakers_detected']}/5 locuteurs détectés")
        else:
            logger.info(f"  - Diarisation: ⚠ {verbatim_metrics['speakers_detected']}/5 locuteurs détectés")
    
    # Sauvegarde du rapport
    report = {
        'timestamp': datetime.now().isoformat(),
        'our_system': our_system,
        'verbatim_system': verbatim_system,
        'file_tested': 'gros_adaptive_enhanced.mp3'
    }
    
    with open('verbatim_comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info("\nRapport sauvegardé dans 'verbatim_comparison_report.json'")

def main():
    """
    Fonction principale du test comparatif
    """
    logger.info("=== TEST COMPARATIF VERBATIM CLI vs NOTRE SYSTÈME ===")
    logger.info("Fichier: gros_adaptive_enhanced.mp3")
    logger.info("Configuration: 5 locuteurs, FR+EN, Faster Whisper Large-V3")
    
    # 1. Exécution de Verbatim
    verbatim_result = run_verbatim_cli()
    
    # 2. Analyse des résultats
    verbatim_metrics = None
    if verbatim_result:
        verbatim_metrics = analyze_verbatim_output(verbatim_result)
    
    # 3. Comparaison avec notre système
    compare_systems(verbatim_metrics)
    
    logger.info("=== TEST TERMINÉ ===")

if __name__ == "__main__":
    main() 