"""
Submit training job to Vertex AI.

Usage:
    python submit_vertex_training.py --config config/model/medium.yaml --machine-type n1-standard-16
"""

import argparse
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform_v1.types.custom_job import Scheduling
import yaml
from pathlib import Path
from datetime import datetime


def upload_config_to_gcs(config_path: str, bucket_name: str, project_id: str) -> str:
    """Upload config file to GCS and return the GCS path."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_name = f"configs/{timestamp}_{Path(config_path).name}"
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(config_path)
    gcs_path = f"gs://{bucket_name}/{blob_name}"
    print(f"Config uploaded to: {gcs_path}")
    return gcs_path


def upload_directory_to_gcs(
    local_dir: str, 
    bucket_name: str, 
    gcs_prefix: str, 
    project_id: str
) -> None:
    """
    Upload an entire directory to GCS, preserving structure.
    
    Args:
        local_dir: Local directory path to upload
        bucket_name: GCS bucket name
        gcs_prefix: Prefix in GCS (e.g., "models/20251117_214059")
        project_id: GCP project ID
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise ValueError(f"Local directory does not exist: {local_dir}")
    
    if not local_path.is_dir():
        raise ValueError(f"Path is not a directory: {local_dir}")
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Get all files recursively
    files_to_upload = list(local_path.rglob("*"))
    files_to_upload = [f for f in files_to_upload if f.is_file()]
    
    if not files_to_upload:
        print(f"Warning: No files found in {local_dir}")
        return
    
    print(f"\nUploading {len(files_to_upload)} files from {local_dir} to gs://{bucket_name}/{gcs_prefix}")
    
    uploaded = 0
    for local_file in files_to_upload:
        # Get relative path from local_dir
        relative_path = local_file.relative_to(local_path)
        # Construct GCS path
        blob_name = f"{gcs_prefix}/{relative_path}".replace("\\", "/")  # Handle Windows paths
        
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_file))
        uploaded += 1
        
        # Progress indicator
        if uploaded % 10 == 0 or uploaded == len(files_to_upload):
            print(f"  Uploaded {uploaded}/{len(files_to_upload)} files...", end="\r")
    
    print(f"\n✓ Successfully uploaded {uploaded} files to gs://{bucket_name}/{gcs_prefix}")


def update_config_for_gcs(
    config_path: str, 
    bucket_name: str, 
    timestamp: str,
    resume_from_local: bool = False
) -> dict:
    """Update config to use /gcs/ mounted paths for Vertex AI."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update data paths to /gcs/ mounted paths
    # Convert relative paths like "data/main/7_datasets/train.pt" 
    # to "/gcs/bucket-name/data/main/7_datasets/train.pt"
    config["data"]["train_path"] = f"/gcs/{bucket_name}/{config['data']['train_path']}"
    config["data"]["val_path"] = f"/gcs/{bucket_name}/{config['data']['val_path']}"
    if "test_path" in config["data"]:
        config["data"]["test_path"] = f"/gcs/{bucket_name}/{config['data']['test_path']}"

    # Update output directory to /gcs/ mounted path
    config["training"]["output_dir"] = f"/gcs/{bucket_name}/models/{timestamp}"

    # Fix resume_from_checkpoint path
    if "resume_from_checkpoint" in config["training"]:
        resume_path = config["training"]["resume_from_checkpoint"]
        if resume_path:
            if resume_from_local:
                # When resuming from uploaded local directory, checkpoint will be in the output dir
                config["training"]["resume_from_checkpoint"] = (
                    f"{config['training']['output_dir']}/checkpoints/latest.pt"
                )
                print(f"Resuming from uploaded checkpoint: {config['training']['resume_from_checkpoint']}")
            elif not resume_path.startswith("/gcs/"):
                # Convert relative path to /gcs/ mounted path using the output directory
                config["training"]["resume_from_checkpoint"] = (
                    f"{config['training']['output_dir']}/checkpoints/latest.pt"
                )
                print(f"Updated resume_from_checkpoint to: {config['training']['resume_from_checkpoint']}")

    return config


def submit_training_job(
    project_id: str,
    region: str,
    bucket_name: str,
    config_path: str,
    image_uri: str,
    machine_type: str = "n1-standard-16",
    accelerator_type: str = "NVIDIA_TESLA_V100",
    accelerator_count: int = 4,
    replica_count: int = 1,
    display_name: str = None,
    use_spot: bool = False,
    resume_from_local: str = None,
):
    """
    Submit a custom training job to Vertex AI.
    
    Args:
        resume_from_local: Optional path to local output directory to upload before training.
                          Should contain checkpoints/ and logs/ subdirectories.
    """

    # Initialize Vertex AI with staging bucket
    staging_bucket = f"gs://{bucket_name}"
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
    )

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Upload local output directory if specified
    if resume_from_local:
        print("\n" + "=" * 80)
        print("Uploading local output directory to GCS")
        print("=" * 80)
        upload_directory_to_gcs(
            local_dir=resume_from_local,
            bucket_name=bucket_name,
            gcs_prefix=f"models/{timestamp}",
            project_id=project_id
        )
        print("=" * 80 + "\n")

    # Update config for /gcs/ mounted paths
    config = update_config_for_gcs(
        config_path, 
        bucket_name, 
        timestamp,
        resume_from_local=resume_from_local is not None
    )

    # Save updated config
    temp_config_path = "/tmp/vertex_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Upload to GCS
    gcs_config_path = upload_config_to_gcs(temp_config_path, bucket_name, project_id)

    # Generate job display name
    if display_name is None:
        suffix = "resume" if resume_from_local else "train"
        display_name = f"saxgpt-{suffix}-{timestamp}"

    # Build machine spec based on accelerator settings
    machine_spec = {"machine_type": machine_type}

    # Only add accelerator if count > 0
    if accelerator_count > 0:
        machine_spec["accelerator_type"] = accelerator_type
        machine_spec["accelerator_count"] = accelerator_count

    # Convert gs:// to /gcs/ for the config path
    config_path_in_container = "/gcs/" + gcs_config_path[5:]
    
    args = [
        "--config",
        config_path_in_container,
    ]
    if accelerator_count > 1:
        args.append("--ddp")
        
    # Worker pool specification
    if replica_count > 1:
        # Distributed training
        worker_pool_specs = [
            {
                "machine_spec": machine_spec,
                "replica_count": replica_count,
                "container_spec": {
                    "image_uri": image_uri,
                    "args": args,
                },
            }
        ]

        print("\nSubmitting DISTRIBUTED training job:")
        print(f"  Replicas: {replica_count}")
        print(f"  Machine type: {machine_type}")
        print(f"  Spot/Preemptible: {use_spot}")
        if accelerator_count > 0:
            print(f"  Accelerator: {accelerator_type} x{accelerator_count} per replica")
            print(f"  Total GPUs: {replica_count * accelerator_count}")
        else:
            print("  Accelerator: None (CPU only)")

    else:
        # Single machine training
        worker_pool_specs = [
            {
                "machine_spec": machine_spec,
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "args": args,
                },
            }
        ]

        print("\nSubmitting SINGLE-MACHINE training job:")
        print(f"  Machine type: {machine_type}")
        print(f"  Spot/Preemptible: {use_spot}")
        if accelerator_count > 0:
            print(f"  Accelerator: {accelerator_type} x{accelerator_count}")
        else:
            print("  Accelerator: None (CPU only)")

    print(f"  Image: {image_uri}")
    print(f"  Config (uploaded): {gcs_config_path}")
    print(f"  Config (in container): {config_path_in_container}")
    print(f"  Output: {config['training']['output_dir']}")
    print(f"  Output (GCS): gs://{bucket_name}/models/{timestamp}")
    if resume_from_local:
        print(f"  Resuming from: {resume_from_local} (uploaded)")

    # Create custom job
    print(f"\nCreating custom job: {display_name}")
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"gs://{bucket_name}/models/{timestamp}",  # Use gs:// for Vertex AI API
    )

    # Submit job
    print("\nSubmitting job...")
    job.submit(
        restart_job_on_worker_restart=True,
        scheduling_strategy=Scheduling.Strategy.SPOT if use_spot else None,
    )

    print("Job submitted successfully.")
    print(f"  Job name: {display_name}")
    print(f"  Job resource name: {job.resource_name}")
    job_id = job.resource_name.split("/")[-1]
    print(f"  Job id: {job_id}")

    print("\nMonitor your job:")
    print(
        f"  Dashboard: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}"
    )
    print("\nStream logs (once job starts):")
    print(f"  gcloud ai custom-jobs stream-logs {job_id} --region={region}")
    print("\nDownload checkpoints (after training):")
    print(f"  gsutil -m cp -r gs://{bucket_name}/models/{timestamp}/checkpoints ./")
    print("\nDownload logs:")
    print(f"  gsutil -m cp -r gs://{bucket_name}/models/{timestamp}/logs ./")

    return job


def main():
    parser = argparse.ArgumentParser(description="Submit training job to Vertex AI")

    # Required arguments
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument(
        "--bucket", required=True, help="GCS bucket name (without gs://)"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--image-uri", required=True, help="Container image URI")

    # Machine configuration
    parser.add_argument(
        "--machine-type",
        default="n1-standard-16",
        help="Machine type (e.g., n1-standard-16, n1-highmem-32)",
    )
    parser.add_argument(
        "--accelerator-type",
        default="NVIDIA_TESLA_V100",
        help="GPU type (e.g., NVIDIA_TESLA_V100, NVIDIA_TESLA_T4, NVIDIA_TESLA_A100)",
    )
    parser.add_argument(
        "--accelerator-count", type=int, default=4, help="Number of GPUs per machine"
    )
    parser.add_argument(
        "--replica-count",
        type=int,
        default=1,
        help="Number of replicas for distributed training",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Use Spot/Preemptible VMs",
    )

    parser.add_argument(
        "--resume-from-local",
        type=str,
        default=None,
        help="Path to local output directory to upload (should contain checkpoints/ and logs/)",
    )

    # Optional
    parser.add_argument("--display-name", default=None, help="Job display name")

    args = parser.parse_args()

    submit_training_job(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket,
        config_path=args.config,
        image_uri=args.image_uri,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        replica_count=args.replica_count,
        display_name=args.display_name,
        use_spot=args.spot,
        resume_from_local=args.resume_from_local,
    )


if __name__ == "__main__":
    main()
