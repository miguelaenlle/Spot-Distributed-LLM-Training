"""Local orchestrator for Phase 1a spot-training experiments.

Runs on your machine, drives AWS via boto3, and runs two experiments:

  - ``baseline`` — one on-demand GPU trains NanoGPT for a fixed budget, reports
    eval metrics;
  - ``spot`` — a spot GPU trains for one segment, is killed, and a second spot
    instance resumes from S3 and finishes — demonstrating survival of a kill.

All AWS calls are isolated in :mod:`orchestrator.aws` and honor ``--dry-run``.
Credentials come from the ambient environment/profile; nothing here reads them.
"""

__version__ = "0.0.1"
