#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List
import logging
import time
import re
from difflib import SequenceMatcher
import json
from transcription import transcribe_audio_improved

def split_into_sentences(text: str) -> List[str]:
    """
    Découpe un texte en phrases.
    """
    text = text.strip()
    sentences = re.split('[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def find_unique_sentences(text1: str, text2: str, similarity_threshold: float = 0.6) -> Tuple[List[str], List[str]]:
    """
    Trouve les phrases uniques dans chaque texte.
    
    Args:
        text1: Premier texte
        text2: Deuxième texte
        similarity_threshold: Seuil de similarité (0-1) pour considérer deux phrases comme différentes
    
    Returns:
        Tuple contenant les listes des phrases uniques pour chaque texte
    """
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)
    
    unique_to_1 = []
    unique_to_2 = []
    
    # Pour chaque phrase dans le premier texte
    for s1 in sentences1:
        is_unique = True
        for s2 in sentences2:
            similarity = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
            if similarity > similarity_threshold:
                is_unique = False
                break
        if is_unique and len(s1.split()) > 3:  # Ignore les phrases trop courtes
            unique_to_1.append(s1)
    
    # Pour chaque phrase dans le deuxième texte
    for s2 in sentences2:
        is_unique = True
        for s1 in sentences1:
            similarity = SequenceMatcher(None, s2.lower(), s1.lower()).ratio()
            if similarity > similarity_threshold:
                is_unique = False
                break
        if is_unique and len(s2.split()) > 3:  # Ignore les phrases trop courtes
            unique_to_2.append(s2)
    
    return unique_to_1, unique_to_2

def analyze_confidence_distribution(segments):
    """
    Analyse la distribution des valeurs de confiance dans les segments.
    Retourne None pour toutes les métriques si aucun segment n'est présent.
    """
    if not segments:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None
        }
    
    confidences = [s["confidence"] for s in segments]
    return {
        "min": min(confidences),
        "max": max(confidences),
        "mean": np.mean(confidences),
        "std": np.std(confidences)
    }

def analyze_transcriptions(original_meta: Dict, enhanced_meta: Dict) -> Dict:
    """
    Analyse les différences entre les transcriptions.
    """
    # Vérification et extraction des segments
    original_segments = original_meta.get("segments", [])
    enhanced_segments = enhanced_meta.get("segments", [])
    
    metrics = {
        "original_duration": original_meta.get("duration", 0),
        "enhanced_duration": enhanced_meta.get("duration", 0),
        "original_segments": len(original_segments),
        "enhanced_segments": len(enhanced_segments),
        "processing_times": {
            "original": original_meta.get("processing_time", 0),
            "enhanced": enhanced_meta.get("processing_time", 0)
        },
        "original_segments_details": original_segments,
        "enhanced_segments_details": enhanced_segments
    }
    
    # Calcul des confiances moyennes
    if original_segments and enhanced_segments:
        original_confidences = [s.get("confidence", 0) for s in original_segments]
        enhanced_confidences = [s.get("confidence", 0) for s in enhanced_segments]
        
        if original_confidences and enhanced_confidences:
            metrics["original_confidence"] = float(np.mean(original_confidences))
            metrics["enhanced_confidence"] = float(np.mean(enhanced_confidences))
            
            # Calcul du pourcentage d'amélioration de la confiance
            confidence_improvement = ((metrics["enhanced_confidence"] - metrics["original_confidence"]) 
                                    / abs(metrics["original_confidence"])) * 100 if metrics["original_confidence"] != 0 else 0
            metrics["confidence_improvement_percent"] = float(confidence_improvement)
            
            # Analyse des distributions de confiance
            metrics["confidence_stats"] = {
                "original": {
                    "min": float(np.min(original_confidences)),
                    "max": float(np.max(original_confidences)),
                    "mean": float(np.mean(original_confidences)),
                    "std": float(np.std(original_confidences))
                },
                "enhanced": {
                    "min": float(np.min(enhanced_confidences)),
                    "max": float(np.max(enhanced_confidences)),
                    "mean": float(np.mean(enhanced_confidences)),
                    "std": float(np.std(enhanced_confidences))
                }
            }
    
    return metrics

def format_text_with_linebreaks(text: str) -> str:
    """Formate le texte avec des sauts de ligne après chaque phrase."""
    if not text.strip():
        return "Aucun texte détecté"
    text = re.sub(r'\.\s+', '.\n', text)
    text = re.sub(r'\?\s+', '?\n', text)
    text = re.sub(r'!\s+', '!\n', text)
    return text

def format_value(value, format_str=".2f"):
    """Formate une valeur en gérant le cas None."""
    if value is None:
        return "N/A"
    try:
        return format(value, format_str)
    except (ValueError, TypeError):
        return str(value)

def save_results(original_text: str, enhanced_text: str, metrics: Dict, output_dir: str):
    """
    Sauvegarde les résultats dans un fichier unique avec les indices de confiance.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde des transcriptions avec indices de confiance
    with open(output_path / "transcription_comparison.txt", "w", encoding="utf-8") as f:
        f.write("TRANSCRIPTION ORIGINALE :\n")
        f.write("=======================\n")
        for segment in metrics.get('original_segments_details', []):
            confidence = segment.get('confidence', 0)
            start_time = format_value(segment.get('start', 0), '.1f')
            end_time = format_value(segment.get('end', 0), '.1f')
            f.write(f"[{start_time}s -> {end_time}s] (confiance: {format_value(confidence, '.4f')})\n")
            f.write(f"{segment.get('text', '')}\n\n")
        
        f.write("\nTRANSCRIPTION AMÉLIORÉE :\n")
        f.write("=======================\n")
        for segment in metrics.get('enhanced_segments_details', []):
            confidence = segment.get('confidence', 0)
            start_time = format_value(segment.get('start', 0), '.1f')
            end_time = format_value(segment.get('end', 0), '.1f')
            f.write(f"[{start_time}s -> {end_time}s] (confiance: {format_value(confidence, '.4f')})\n")
            f.write(f"{segment.get('text', '')}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Évalue l'amélioration de la qualité audio")
    parser.add_argument('--original', '-o', required=True, help='Chemin vers le fichier audio original')
    parser.add_argument('--enhanced', '-e', required=True, help='Chemin vers le fichier audio amélioré')
    parser.add_argument('--output-dir', '-d', default='evaluation_results',
                       help='Dossier de sortie pour les résultats')
    parser.add_argument('--verbose', '-v', action='store_true', help='Afficher les messages de débogage')
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Transcription des deux fichiers
        start_time = time.time()
        original_text, original_meta = transcribe_audio_improved(args.original)
        original_meta['processing_time'] = time.time() - start_time
        
        start_time = time.time()
        enhanced_text, enhanced_meta = transcribe_audio_improved(args.enhanced)
        enhanced_meta['processing_time'] = time.time() - start_time
        
        # Debug des métadonnées
        logger.debug("Métadonnées originales :")
        logger.debug(json.dumps(original_meta, indent=2))
        logger.debug("\nMétadonnées améliorées :")
        logger.debug(json.dumps(enhanced_meta, indent=2))
        
        # Analyse des différences
        metrics = analyze_transcriptions(original_meta, enhanced_meta)
        
        # Sauvegarde des résultats
        save_results(original_text, enhanced_text, metrics, args.output_dir)
        
        # Affichage des métriques principales
        logger.info("\nRésultats de l'évaluation :")
        logger.info(f"Nombre de segments originaux : {metrics['original_segments']}")
        logger.info(f"Nombre de segments améliorés : {metrics['enhanced_segments']}")
        
        if 'confidence_improvement_percent' in metrics:
            logger.info(f"Amélioration de la confiance : {format_value(metrics['confidence_improvement_percent'])}%")
        
        if 'confidence_stats' in metrics:
            logger.info("\nStatistiques de confiance :")
            logger.info("Original - Moyenne : {}, Écart-type : {}".format(
                format_value(metrics['confidence_stats']['original']['mean']),
                format_value(metrics['confidence_stats']['original']['std'])
            ))
            logger.info("Amélioré - Moyenne : {}, Écart-type : {}".format(
                format_value(metrics['confidence_stats']['enhanced']['mean']),
                format_value(metrics['confidence_stats']['enhanced']['std'])
            ))
        
        logger.info(f"\nRésultats détaillés sauvegardés dans : {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 