"""Inference fleet (ROADMAP Part 1) — serve trained checkpoints behind one API.

A router load-balances over N workers; workers heartbeat to the store
(S3 or a local dir) and the router reroutes failed requests, so a spot
worker dying mid-request is invisible to the client.
"""
