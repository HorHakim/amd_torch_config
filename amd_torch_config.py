import torch
import os
import warnings
from typing import Dict

def configure_torch() -> Dict[str, bool]:
    """
    Configure PyTorch pour une utilisation optimale avec ROCm sur GPU AMD.
    
    Cette fonction:
    1. Gère les warnings et messages d'erreur
       - Supprime le warning hipBLASLt qui n'est pas pertinent pour RDNA 2
    
    2. Configure les variables d'environnement ROCm
       - Désactive hipBLASLt qui n'est pas optimisé pour RDNA 2
       - Force l'utilisation de ROCBLAS pour de meilleures performances
    
    3. Optimise les performances CUDNN
       - Active l'autotuning pour trouver les meilleurs algorithmes
       - Désactive le mode déterministe pour plus de performance
       - Active CUDNN pour les opérations optimisées
    
    Returns:
        Dict[str, bool]: État des différentes configurations
    """
    config_status = {}
    
    # 1. Gestion des warnings
    warnings.filterwarnings('ignore', message='Attempting to use hipBLASLt')
    config_status['warnings_configured'] = True
    
    # 2. Configuration ROCm
    os.environ['PYTORCH_HIPBLASLT_DISABLE'] = '1' # Désactive hipBLASLt qui cause des warnings et n'est pas optimisé pour RDNA 2
    os.environ['TORCH_USE_ROCBLAS'] = '1' # Force l'utilisation de ROCBLAS, qui est plus stable et mieux optimisé pour votre GPU
    config_status['rocm_configured'] = True
    
    # 3. Optimisation des performances
    # Active l'autotuning : PyTorch va tester différents algorithmes et choisir le plus rapide
    # Très utile si vous utilisez toujours les mêmes tailles de tenseurs (comme dans le CNN)
    torch.backends.cudnn.benchmark = True     # Autotuning


    # Permet des optimisations non-déterministes
    # Améliore les performances mais les résultats peuvent varier légèrement entre les exécutions
    torch.backends.cudnn.deterministic = False  # Mode non-déterministe pour plus de performances


   # Active les optimisations CUDNN pour les opérations comme les convolutions
   # Crucial pour les performances des réseaux de neurones
    torch.backends.cudnn.enabled = True       # Active CUDNN
    config_status['performance_configured'] = True

    return config_status