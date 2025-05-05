# Test Amélioration Audio et Transcription

## Description

Ce projet vise à améliorer la qualité des enregistrements audio et leur transcription en utilisant diverses techniques de traitement du signal et de reconnaissance vocale. Il se compose de deux parties principales :

1. L'amélioration de la qualité audio
2. La transcription et l'analyse du contenu

## Choix Techniques et Paramètres

### 1. Chaîne de Traitement Audio

#### a) Réduction du bruit par soustraction spectrale
**Problème** : Les enregistrements contiennent souvent des bruits de fond (ventilateur, climatisation, rumeur...)
**Solution** : La soustraction spectrale analyse le "profil" du bruit et le retire intelligemment
**Paramètres** :
- Facteur de réduction : 1.8
  - En dessous de 1.5 : bruit encore perceptible
  - Au-dessus de 2.0 : risque d'artefacts audio
  - 1.8 : meilleur compromis qualité/artefacts
**Bénéfice** : Voix plus claire et plus facile à comprendre, comme si on "nettoyait" l'audio

#### b) Amplification multi-niveaux adaptative
**Problème** : Certaines parties de la parole sont trop faibles (murmures, fins de phrases...)
**Solution et Paramètres** : 
- Niveau 1 : Sons faibles
  - Seuil : -28dB (niveau typique des fins de phrases)
  - Amplification x3.5 (optimal sans distorsion)
- Niveau 2 : Sons très faibles
  - Seuil : -38dB (niveau des murmures)
  - Amplification x5.0 (maximum avant distorsion)
**Bénéfice** : Les parties faibles deviennent audibles sans sur-amplifier les parties déjà fortes

#### c) Compression dynamique
**Problème** : Trop grand écart entre les sons forts et faibles
**Solution et Paramètres** : 
- Seuil : -30dB (préserve la dynamique naturelle)
- Ratio : 3:1
  - Réduit les écarts sans "écraser" le son
  - Au-delà : risque de son artificiel
**Bénéfice** : Volume plus uniforme et confortable à l'écoute

#### d) Amélioration de la clarté vocale
**Problème** : La voix peut manquer de présence ou être masquée
**Solution et Paramètres** : 
- Bande de fréquences : 2-4 kHz (zone optimale des fréquences vocales)
- Gain : 0.4
  - En dessous : effet peu perceptible
  - Au-dessus : risque de sifflement
**Bénéfice** : Meilleure intelligibilité des mots, comme un "focus" sur la voix

#### e) Normalisation du signal
**Problème** : Le volume global peut être trop faible ou trop fort
**Solution** : Ajustement automatique du volume à un niveau optimal
**Bénéfice** : Volume final cohérent et agréable à l'écoute

### 2. Système de Transcription et Analyse

#### a) Transcription avec Whisper
**Problème** : Besoin de convertir la parole en texte avec précision
**Solution et Paramètres** : 
- Modèle : "large-v3"
  - Meilleure précision pour les voix faibles
  - Optimisations matérielles :
    - Calcul en float16 sur GPU, int8 sur CPU
    - 4 threads CPU pour le traitement
    - 2 workers pour le parallélisme
- Configuration de la transcription :
  - Langue : français (forcé)
  - Beam size : 5 (recherche élargie des hypothèses)
  - Patience : 2.0 (plus tolérant sur les segments difficiles)
  - Temperature : 0.0 (pas d'aléatoire, résultats déterministes)
  - Best of : 5 (sélection de la meilleure parmi 5 transcriptions)
  - Contexte : utilisation du texte précédent
- Détection de la voix (VAD) :
  - Silence minimum : 250ms
  - Padding voix : 150ms (contexte élargi)
  - Seuil : 0.30 (adapté aux voix faibles)
**Bénéfice** : Transcription haute précision, particulièrement adaptée aux voix faibles ou peu audibles

#### b) Segmentation intelligente
**Problème** : Les longues transcriptions en bloc sont difficiles à analyser
**Solution et Paramètres** : 
- Pauses naturelles
  - Seuil : 0.4s (optimal)
  - < 0.3s : découpage trop agressif
  - > 0.5s : segments trop longs
- Seuil de confiance : 0.6
  - Balance précision/sur-segmentation
**Bénéfice** : Texte plus facile à lire et à analyser

#### c) Analyse de confiance

**Problème** : Besoin de savoir quelles parties de la transcription sont fiables
**Solution** : Calcul d'un score de confiance pour chaque segment et chaque mot

- Score par mot : précision de la reconnaissance
- Score par segment : moyenne pondérée des mots
  **Bénéfice** : Identification rapide des passages potentiellement incorrects

#### d) Comparaison avant/après

**Problème** : Comment mesurer l'impact de l'amélioration audio ?
**Solution** : Analyse comparative automatique :

- Nombre de segments détectés
- Scores de confiance moyens
- Différences de transcription
  **Bénéfice** : Évaluation objective de l'amélioration obtenue

#### e) Export JSON structuré

**Problème** : Besoin d'exploiter les résultats dans d'autres systèmes
**Solution** : Format JSON avec métadonnées détaillées :

- Timestamps précis (début/fin de chaque mot)
- Scores de confiance à tous les niveaux
- Informations sur la langue et la durée
  **Bénéfice** : Facilite l'intégration avec d'autres outils et analyses

### Résultat global de la transcription

La chaîne de traitement permet :

1. Une transcription de qualité professionnelle
2. Un découpage intelligent du texte
3. Une évaluation précise de la fiabilité
4. Une comparaison objective des améliorations
5. Une exploitation facile des résultats

Les paramètres ont été choisis pour maximiser la précision tout en gardant des temps de traitement raisonnables.

## Résultats d'Évaluation

### Métriques Clés

- **Segmentation** :
  - Version originale : 1 segment
  - Version améliorée : 9 segments
- **Scores de confiance** :
  - Original : -0.2485 (moyenne)
  - Amélioré : entre -0.3258 et -0.2200
- **Durée analysée** : 43.4 secondes

### Exemple de Transcription

#### Version Originale
Un seul segment long (15.1s -> 35.1s) :
```
Attention aux jours fériés là, on n'a rien. Il y a un jour férié là, un lundi de Pentecôte. D'accord, donc il faudrait peut-être mieux, parce qu'avec un truc un peu agruyère là, c'est jamais très... On bascule en début juin.
```

#### Version Améliorée
Segments plus courts et précis, par exemple :
```
[0.0s -> 11.1s] "Vous pouvez y aller. Du coup, j'ai besoin que l'on confirme la date de l'événement..."
[14.9s -> 17.6s] "Attention au jour férié là, on n'a rien..."
```

Cette segmentation plus fine permet :
- Une meilleure compréhension du contexte
- Une identification précise des interlocuteurs
- Un repérage facile des points importants

### Améliorations Constatées

1. **Segmentation plus fine**

   - Meilleure identification des pauses naturelles
   - Découpage plus précis des phrases
   - Facilite l'analyse du discours
2. **Précision temporelle**

   - Timestamps précis pour chaque segment
   - Identification exacte des mots et leurs durées
   - Meilleure synchronisation avec l'audio
3. **Analyse détaillée**

   - Score de confiance par segment
   - Identification des zones problématiques
   - Possibilité de traitement ciblé

## Limitations et Perspectives

- Scores de confiance légèrement plus bas sur certains segments courts
- Possibilité d'amélioration par :
  - Ajustement des paramètres de segmentation
  - Optimisation des algorithmes d'amélioration audio
  - Calibration des seuils de confiance

## Utilisation

### Amélioration et Transcription

```bash
python audio_enhancer.py --input fichier.wav --transcribe --export-json
```

### Évaluation

```bash
python evaluate_enhancement.py --original fichier.wav --enhanced fichier_enhanced.wav
```

## Dépendances

- Python 3.x
- NumPy
- SciPy
- Whisper
- librosa
- soundfile

## Paramètres Techniques : Choix et Justifications

### Amélioration Audio

#### Réduction de bruit

- **Facteur = 1.8**
  - En dessous de 1.5 : bruit encore perceptible
  - Au-dessus de 2.0 : risque d'artefacts audio
  - 1.8 : meilleur compromis qualité/artefacts

#### Amplification multi-niveaux

- **Seuil faible = -28dB, boost x3.5**
  - -28dB : niveau typique des fins de phrases
  - x3.5 : amplifie sans créer de distorsion
- **Seuil très faible = -38dB, boost x5.0**
  - -38dB : niveau des murmures/bruits de fond
  - x5.0 : maximum avant distorsion du signal

#### Compression dynamique

- **Seuil = -30dB, ratio = 3:1**
  - -30dB : préserve la dynamique naturelle
  - Ratio 3:1 : réduit les écarts sans "écraser" le son

#### Clarté vocale

- **Bande = 2-4 kHz, gain = 0.4**
  - 2-4 kHz : fréquences essentielles de la voix
  - Gain 0.4 : renforce sans créer de sifflement

### Transcription

#### Modèle Whisper

- **Version "large-v3"**
  - Modèle le plus précis de la gamme Whisper
  - Paramètres optimisés pour la qualité :
    - Recherche approfondie (beam=5, best_of=5)
    - Mode déterministe (temp=0)
    - Utilisation du contexte
  - Configuration VAD fine :
    - Détection sensible (seuil 0.30)
    - Segmentation précise (250ms/150ms)
  - Optimisations matérielles :
    - GPU : float16
    - CPU : int8
    - Multi-threading : 4/2

#### Segmentation

- **Pause minimale = 0.4s**
  - < 0.3s : découpage trop agressif
  - > 0.5s : segments trop longs
    >
- **Seuil de confiance = 0.6**
  - Balance entre précision et sur-segmentation

Ces paramètres ont été déterminés par tests successifs sur un échantillon représentatif d'enregistrements, en privilégiant la qualité du résultat final tout en évitant les effets indésirables.
