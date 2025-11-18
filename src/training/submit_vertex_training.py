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
from time import sleep


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


def update_config_for_gcs(config_path: str, bucket_name: str) -> dict:
    """Update config to use GCS paths."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update data paths to GCS
    config["data"]["train_path"] = f"gs://{bucket_name}/{config['data']['train_path']}"
    config["data"]["val_path"] = f"gs://{bucket_name}/{config['data']['val_path']}"
    if "test_path" in config["data"]:
        config["data"]["test_path"] = (
            f"gs://{bucket_name}/{config['data']['test_path']}"
        )

    # Update output directory to GCS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["training"]["output_dir"] = f"gs://{bucket_name}/models/{timestamp}"

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
):
    """Submit a custom training job to Vertex AI."""

    # Initialize Vertex AI with staging bucket
    staging_bucket = f"gs://{bucket_name}"
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
    )

    # Update config for GCS
    config = update_config_for_gcs(config_path, bucket_name)

    # Save updated config
    temp_config_path = "/tmp/vertex_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Upload to GCS
    gcs_config_path = upload_config_to_gcs(temp_config_path, bucket_name, project_id)

    # Generate job display name
    if display_name is None:
        display_name = f"saxgpt-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build machine spec based on accelerator settings
    machine_spec = {"machine_type": machine_type}

    # Only add accelerator if count > 0
    if accelerator_count > 0:
        machine_spec["accelerator_type"] = accelerator_type
        machine_spec["accelerator_count"] = accelerator_count

    args = [
        "--config",
        "/gcs" + gcs_config_path[4:],  # Remove 'gs:/' prefix
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
    print(f"  Config: {gcs_config_path}")
    print(f"  Output: {config['training']['output_dir']}")

    # Create custom job
    print(f"\nCreating custom job: {display_name}")
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=config["training"]["output_dir"],
    )

    # Submit job
    print("\nSubmitting job...")
    job.run(
        sync=False,
        restart_job_on_worker_restart=True,
        scheduling_strategy=Scheduling.Strategy.SPOT if use_spot else None,
    )

    print("Job submitted successfully.")
    print(f"  Job name: {display_name}")
    print("Getting job id...")

    try:
        while True:
            try:
                print(f"  Job resource name: {job.resource_name}")
                job_id = job.resource_name.split("/")[-1]
                print(f"  Job id: {job_id}")
                break
            except RuntimeError:
                sleep(5.0)
    except KeyboardInterrupt:
        # Show quota errors with synchronous call
        job.run(
            sync=True,
            restart_job_on_worker_restart=True,
            scheduling_strategy=Scheduling.Strategy.SPOT if use_spot else None,
        )

    print("\nMonitor your job:")
    print(
        f"  Dashboard: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}"
    )
    print("\nStream logs (once job starts):")
    print(f"  gcloud ai custom-jobs stream-logs {job_id} --region={region}")
    print("\n Download checkpoints (after training):")
    print(f"  gsutil -m cp -r {config['training']['output_dir']}/checkpoints ./")

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

    # Distributed training
    parser.add_argument(
        "--replica-count",
        type=int,
        default=1,
        help="Number of replicas for distributed training",
    )

    # Cost optimization
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Use Spot/Preemptible VMs (up to 80%% cheaper, can be interrupted)",
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
    )


if __name__ == "__main__":
    main()
