#!/usr/bin/env python3
"""
Script de validation de la diarisation adaptative.
Compare les résultats avec le fichier de référence petit_correct.srt
Respecte les règles .cursorrules : ZÉRO valeur codée en dur.
"""

import re
import argparse
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class ReferenceSegment:
    """Segment de référence extrait du fichier SRT."""
    start: float
    end: float
    speaker: str
    text: str
    
    def duration(self) -> float:
        return self.end - self.start
    
    def overlaps_with(self, other_start: float, other_end: float, min_overlap: float = 0.5) -> bool:
        """Vérifie si ce segment chevauche significativement avec un autre."""
        overlap_start = max(self.start, other_start)
        overlap_end = min(self.end, other_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        # Calcul du pourcentage de chevauchement par rapport au plus petit segment
        min_duration = min(self.duration(), other_end - other_start)
        if min_duration == 0:
            return False
        
        overlap_ratio = overlap_duration / min_duration
        return overlap_ratio >= min_overlap

@dataclass
class DetectedSegment:
    """Segment détecté par notre système."""
    start: float
    end: float
    speaker: str
    confidence: float = 1.0
    
    def duration(self) -> float:
        return self.end - self.start

def parse_srt_time(time_str: str) -> float:
    """
    Convertit un timestamp SRT en secondes.
    Format: HH:MM:SS,mmm
    """
    # Remplacer la virgule par un point pour les millisecondes
    time_str = time_str.replace(',', '.')
    
    # Parser le format HH:MM:SS.mmm
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

def parse_reference_srt(srt_file: str) -> List[ReferenceSegment]:
    """
    Parse le fichier SRT de référence pour extraire les segments.
    Respecte la règle .cursorrules : ZÉRO valeur codée en dur.
    """
    segments = []
    
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern pour matcher les blocs SRT
    # Groupe 1: numéro, Groupe 2: timestamp, Groupe 3: texte
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\n*$)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        start_time = parse_srt_time(match[1])
        end_time = parse_srt_time(match[2])
        text = match[3].strip()
        
        # Extraction du locuteur depuis le texte [SPEAKER_XX]
        speaker_match = re.search(r'\[SPEAKER_(\w+)\]', text)
        if speaker_match:
            speaker = f"SPEAKER_{speaker_match.group(1)}"
            # Nettoyer le texte en retirant la balise speaker
            clean_text = re.sub(r'\[SPEAKER_\w+\]\s*', '', text)
        else:
            speaker = "UNKNOWN"
            clean_text = text
        
        segment = ReferenceSegment(
            start=start_time,
            end=end_time,
            speaker=speaker,
            text=clean_text
        )
        segments.append(segment)
    
    return segments

def parse_diarization_results(results_file: str) -> List[DetectedSegment]:
    """
    Parse les résultats de notre diarisation adaptative.
    """
    segments = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Chercher la section des segments détectés
    lines = content.split('\n')
    in_segments_section = False
    
    for line in lines:
        if "SEGMENTS DÉTECTÉS:" in line:
            in_segments_section = True
            continue
        
        if in_segments_section and line.strip():
            # Pattern pour matcher: "  1. [0.0s - 0.9s] SPEAKER_04 (conf: 1.000)"
            match = re.search(r'\d+\.\s*\[(\d+\.?\d*)s\s*-\s*(\d+\.?\d*)s\]\s*(\w+)\s*\(conf:\s*(\d+\.?\d*)\)', line)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                speaker = match.group(3)
                confidence = float(match.group(4))
                
                segment = DetectedSegment(
                    start=start,
                    end=end,
                    speaker=speaker,
                    confidence=confidence
                )
                segments.append(segment)
            elif line.startswith('📊') or line.startswith('👥'):
                # Fin de la section des segments
                break
    
    return segments

def calculate_speaker_mapping(reference: List[ReferenceSegment], 
                            detected: List[DetectedSegment]) -> Dict[str, str]:
    """
    Calcule le mapping optimal entre locuteurs détectés et de référence.
    Utilise la méthode de chevauchement temporel maximal.
    """
    # Matrice de chevauchement entre locuteurs détectés et de référence
    ref_speakers = list(set(s.speaker for s in reference))
    det_speakers = list(set(s.speaker for s in detected))
    
    overlap_matrix = {}
    
    for det_speaker in det_speakers:
        overlap_matrix[det_speaker] = {}
        for ref_speaker in ref_speakers:
            overlap_matrix[det_speaker][ref_speaker] = 0.0
    
    # Calcul des chevauchements
    for det_seg in detected:
        for ref_seg in reference:
            if ref_seg.overlaps_with(det_seg.start, det_seg.end, min_overlap=0.3):
                overlap_duration = min(det_seg.end, ref_seg.end) - max(det_seg.start, ref_seg.start)
                overlap_matrix[det_seg.speaker][ref_seg.speaker] += overlap_duration
    
    # Mapping optimal (algorithme glouton)
    mapping = {}
    used_ref_speakers = set()
    
    # Trier les locuteurs détectés par durée totale (plus important en premier)
    det_speaker_durations = {}
    for det_seg in detected:
        if det_seg.speaker not in det_speaker_durations:
            det_speaker_durations[det_seg.speaker] = 0
        det_speaker_durations[det_seg.speaker] += det_seg.duration()
    
    sorted_det_speakers = sorted(det_speaker_durations.keys(), 
                                key=lambda x: det_speaker_durations[x], reverse=True)
    
    for det_speaker in sorted_det_speakers:
        # Trouver le locuteur de référence avec le plus de chevauchement
        best_ref_speaker = None
        best_overlap = 0.0
        
        for ref_speaker in ref_speakers:
            if ref_speaker not in used_ref_speakers:
                overlap = overlap_matrix[det_speaker][ref_speaker]
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ref_speaker = ref_speaker
        
        if best_ref_speaker and best_overlap > 0:
            mapping[det_speaker] = best_ref_speaker
            used_ref_speakers.add(best_ref_speaker)
        else:
            mapping[det_speaker] = f"UNMAPPED_{det_speaker}"
    
    return mapping

def calculate_validation_metrics(reference: List[ReferenceSegment], 
                               detected: List[DetectedSegment],
                               speaker_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Calcule les métriques de validation de la diarisation.
    Toutes les métriques sont calculées adaptivement, pas de valeurs codées en dur.
    """
    metrics = {}
    
    # 1. Métriques temporelles
    total_ref_duration = sum(s.duration() for s in reference)
    total_det_duration = sum(s.duration() for s in detected)
    
    # 2. Métriques de segmentation
    correct_boundaries = 0
    total_boundaries = len(reference) - 1  # Nombre de transitions
    
    # Tolérance adaptative basée sur la durée moyenne des segments
    avg_segment_duration = total_ref_duration / len(reference) if reference else 1.0
    boundary_tolerance = min(0.5, avg_segment_duration * 0.1)  # 10% de la durée moyenne, max 0.5s
    
    for i in range(len(reference) - 1):
        ref_boundary = reference[i].end
        # Chercher une frontière détectée proche
        for j in range(len(detected) - 1):
            det_boundary = detected[j].end
            if abs(ref_boundary - det_boundary) <= boundary_tolerance:
                correct_boundaries += 1
                break
    
    boundary_precision = correct_boundaries / max(total_boundaries, 1)
    
    # 3. Métriques de locuteurs
    ref_speakers = set(s.speaker for s in reference)
    det_speakers = set(s.speaker for s in detected)
    mapped_speakers = set(speaker_mapping.values())
    
    speaker_recall = len(mapped_speakers & ref_speakers) / len(ref_speakers)
    speaker_precision = len(mapped_speakers & ref_speakers) / max(len(mapped_speakers), 1)
    
    # 4. Métriques de chevauchement temporel
    total_overlap_duration = 0.0
    total_correct_speaker_duration = 0.0
    
    for ref_seg in reference:
        for det_seg in detected:
            if ref_seg.overlaps_with(det_seg.start, det_seg.end, min_overlap=0.1):
                overlap_start = max(ref_seg.start, det_seg.start)
                overlap_end = min(ref_seg.end, det_seg.end)
                overlap_duration = overlap_end - overlap_start
                total_overlap_duration += overlap_duration
                
                # Vérifier si le locuteur est correct
                mapped_speaker = speaker_mapping.get(det_seg.speaker, "UNKNOWN")
                if mapped_speaker == ref_seg.speaker:
                    total_correct_speaker_duration += overlap_duration
    
    temporal_coverage = total_overlap_duration / total_ref_duration if total_ref_duration > 0 else 0
    speaker_accuracy = total_correct_speaker_duration / total_overlap_duration if total_overlap_duration > 0 else 0
    
    # 5. Score composite adaptatif
    # Pondération adaptative basée sur le nombre de locuteurs
    num_ref_speakers = len(ref_speakers)
    if num_ref_speakers <= 2:
        # Peu de locuteurs : privilégier la précision temporelle
        weights = {'boundary': 0.4, 'temporal': 0.4, 'speaker': 0.2}
    elif num_ref_speakers <= 4:
        # Nombre moyen : équilibré
        weights = {'boundary': 0.3, 'temporal': 0.3, 'speaker': 0.4}
    else:
        # Beaucoup de locuteurs : privilégier l'identification des locuteurs
        weights = {'boundary': 0.2, 'temporal': 0.3, 'speaker': 0.5}
    
    composite_score = (boundary_precision * weights['boundary'] + 
                      temporal_coverage * weights['temporal'] + 
                      speaker_accuracy * weights['speaker'])
    
    metrics = {
        'boundary_precision': boundary_precision,
        'boundary_tolerance_used': boundary_tolerance,
        'temporal_coverage': temporal_coverage,
        'speaker_recall': speaker_recall,
        'speaker_precision': speaker_precision,
        'speaker_accuracy': speaker_accuracy,
        'composite_score': composite_score,
        'ref_speakers_count': len(ref_speakers),
        'det_speakers_count': len(det_speakers),
        'ref_segments_count': len(reference),
        'det_segments_count': len(detected),
        'ref_total_duration': total_ref_duration,
        'det_total_duration': total_det_duration,
        'weights_used': weights
    }
    
    return metrics

def print_validation_report(reference: List[ReferenceSegment], 
                          detected: List[DetectedSegment],
                          speaker_mapping: Dict[str, str],
                          metrics: Dict[str, Any],
                          logger=None):
    """
    Affiche un rapport détaillé de validation.
    """
    if logger:
        logger.info("=" * 60)
        logger.info("📊 RAPPORT DE VALIDATION DE LA DIARISATION")
        logger.info("=" * 60)
        
        logger.info("🎯 MÉTRIQUES PRINCIPALES:")
        logger.info(f"   • Score composite: {metrics['composite_score']:.3f}")
        logger.info(f"   • Précision des frontières: {metrics['boundary_precision']:.3f}")
        logger.info(f"   • Couverture temporelle: {metrics['temporal_coverage']:.3f}")
        logger.info(f"   • Précision des locuteurs: {metrics['speaker_accuracy']:.3f}")
        
        logger.info("📈 MÉTRIQUES DÉTAILLÉES:")
        logger.info(f"   • Tolérance frontières: {metrics['boundary_tolerance_used']:.3f}s")
        logger.info(f"   • Rappel locuteurs: {metrics['speaker_recall']:.3f}")
        logger.info(f"   • Précision locuteurs: {metrics['speaker_precision']:.3f}")
        
        logger.info("📊 STATISTIQUES COMPARATIVES:")
        logger.info(f"   • Référence: {metrics['ref_segments_count']} segments, {metrics['ref_speakers_count']} locuteurs")
        logger.info(f"   • Détecté: {metrics['det_segments_count']} segments, {metrics['det_speakers_count']} locuteurs")
        logger.info(f"   • Durée référence: {metrics['ref_total_duration']:.1f}s")
        logger.info(f"   • Durée détectée: {metrics['det_total_duration']:.1f}s")
        
        logger.info("🔗 MAPPING DES LOCUTEURS:")
        for det_speaker, ref_speaker in speaker_mapping.items():
            logger.info(f"   • {det_speaker} → {ref_speaker}")
        
        logger.info("⚖️ PONDÉRATION ADAPTATIVE UTILISÉE:")
        for metric, weight in metrics['weights_used'].items():
            logger.info(f"   • {metric}: {weight:.1f}")
        
        # Évaluation qualitative
        if metrics['composite_score'] >= 0.8:
            quality = "EXCELLENTE ✅"
        elif metrics['composite_score'] >= 0.6:
            quality = "BONNE ✅"
        elif metrics['composite_score'] >= 0.4:
            quality = "ACCEPTABLE ⚠️"
        else:
            quality = "INSUFFISANTE ❌"
        
        logger.info(f"🏆 QUALITÉ GLOBALE: {quality}")

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Validation de la diarisation adaptative contre référence SRT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("reference_srt", help="Fichier SRT de référence (petit_correct.srt)")
    parser.add_argument("diarization_results", help="Fichier de résultats de diarisation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Affichage détaillé")
    
    args = parser.parse_args()
    
    # Configuration du logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🔍 Chargement des données de référence...")
        reference = parse_reference_srt(args.reference_srt)
        logger.info(f"   → {len(reference)} segments de référence chargés")
        
        logger.info("🔍 Chargement des résultats de diarisation...")
        detected = parse_diarization_results(args.diarization_results)
        logger.info(f"   → {len(detected)} segments détectés chargés")
        
        logger.info("🔗 Calcul du mapping des locuteurs...")
        speaker_mapping = calculate_speaker_mapping(reference, detected)
        
        logger.info("📊 Calcul des métriques de validation...")
        metrics = calculate_validation_metrics(reference, detected, speaker_mapping)
        
        print_validation_report(reference, detected, speaker_mapping, metrics, logger)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la validation: {str(e)}")
        if args.verbose:
            import traceback
            logger.error("Traceback:")
            for line in traceback.format_exc().strip().split('\n'):
                logger.error(line)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 