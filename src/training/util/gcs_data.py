"""
Utility functions for handling GCS paths in training.
Automatically downloads data from GCS when needed.
"""

import torch
from pathlib import Path
import os
import tempfile


def is_gcs_path(path: str) -> bool:
    """Check if path is a GCS path."""
    return path.startswith('gs://') or path.startswith('/gcs/')


def download_from_gcs(gcs_path: str, local_path: str = None) -> str:
    """
    Download file from GCS to local path.
    
    Args:
        gcs_path: GCS path (gs://bucket/path or /gcs/bucket/path)
        local_path: Local destination path (None = temp file)
    
    Returns:
        Local path where file was downloaded
    """
    from google.cloud import storage
    
    # Handle /gcs/ prefix (Vertex AI mounted GCS)
    if gcs_path.startswith('/gcs/'):
        # Already mounted, just return the path
        return gcs_path
    
    # Parse GCS path
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    gcs_path = gcs_path[5:]  # Remove 'gs://'
    bucket_name, blob_name = gcs_path.split('/', 1)
    
    # Create local path
    if local_path is None:
        suffix = Path(blob_name).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        local_path = temp_file.name
        temp_file.close()
    else:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    print(f"Downloading gs://{bucket_name}/{blob_name} to {local_path}")
    blob.download_to_filename(local_path)
    
    return local_path


def load_dataset_from_path(path: str):
    """
    Load dataset from local or GCS path.
    
    Args:
        path: Local path or GCS path (gs://... or /gcs/...)
    
    Returns:
        Loaded dataset
    """
    if is_gcs_path(path):
        if path.startswith('/gcs/'):
            # GCS is mounted, load directly
            print(f"Loading dataset from mounted GCS: {path}")
            return torch.load(path, weights_only=False, map_location='cpu')
        else:
            # Download first
            local_path = download_from_gcs(path)
            dataset = torch.load(local_path, weights_only=False, map_location='cpu')
            os.unlink(local_path)
            return dataset
    else:
        # Local path
        return torch.load(path, weights_only=False, map_location='cpu')


def upload_to_gcs(local_path: str, gcs_path: str):
    """
    Upload file to GCS.
    
    Args:
        local_path: Local file path
        gcs_path: Destination GCS path (gs://bucket/path)
    """
    from google.cloud import storage
    
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    gcs_path = gcs_path[5:]  # Remove 'gs://'
    bucket_name, blob_name = gcs_path.split('/', 1)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    print(f"Uploading {local_path} to gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(local_path)


def ensure_gcs_checkpoint_dir(output_dir: str) -> str:
    """
    Ensure checkpoint directory exists (handle GCS paths).
    
    Args:
        output_dir: Output directory path (local or GCS)
    
    Returns:
        Checkpoint directory path
    """
    if is_gcs_path(output_dir):
        # For GCS, just return the path (directories don't need to be created)
        checkpoint_dir = output_dir.rstrip('/') + '/checkpoints'
        return checkpoint_dir
    else:
        # For local paths, create directory
        checkpoint_dir = Path(output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return str(checkpoint_dir)


def save_checkpoint_with_gcs(
    checkpoint_data: dict,
    checkpoint_path: str,
):
    """
    Save checkpoint, handling GCS paths.
    
    Args:
        checkpoint_data: Dictionary with checkpoint data
        checkpoint_path: Destination path (local or GCS)
    """
    if is_gcs_path(checkpoint_path):
        # Save to temp file first
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        temp_path = temp_file.name
        temp_file.close()
        
        torch.save(checkpoint_data, temp_path)
        
        if checkpoint_path.startswith('/gcs/'):
            # Mounted GCS - save directly
            import shutil
            shutil.copy(temp_path, checkpoint_path)
            os.unlink(temp_path)
        else:
            # Upload to GCS
            upload_to_gcs(temp_path, checkpoint_path)
            os.unlink(temp_path)
    else:
        # Local path - save directly
        torch.save(checkpoint_data, checkpoint_path)
