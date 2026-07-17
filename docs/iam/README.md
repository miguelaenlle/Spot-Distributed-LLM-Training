# IAM policies

Three least-privilege policies, split by principal. Replace `YOUR_BUCKET` and
`ACCOUNT_ID` (your 12-digit account number) before attaching.

| Policy | Attach to | When |
|--------|-----------|------|
| [`setup-policy.json`](./setup-policy.json) | the **human** running `setup` (laptop user / SSO session) | one-time; creates the bucket + worker role/profile |
| [`controller-policy.json`](./controller-policy.json) | the **orchestrator** — laptop now, an instance-profile role when it becomes a cloud node | every `stage-data` / `baseline` / `spot` run |
| [`worker-policy.json`](./worker-policy.json) | the **training box** (`spot-train-role` instance profile) | attached automatically by `setup` |

`setup` also attaches the AWS-managed `AmazonSSMManagedInstanceCore` policy to the
worker role so you can attach a shell via SSM Session Manager (no inbound ports)
to watch training live — see the main README's "Watch a run live".

## Design notes

- **Roles over users.** The code (`orchestrator/aws.py`) never references secret
  values — boto3 resolves creds from its provider chain (env → profile → …→
  instance metadata). So the *same* controller policy works whether the
  orchestrator runs on your laptop (SSO-assumed role or user keys) or later on an
  EC2 node (attached instance-profile role, no keys at all).
- **`setup` is a one-time human action.** It needs `iam:CreateRole` etc., which
  you should **not** grant to an automated cloud controller (too large a blast
  radius). Run `setup` once from your laptop with `setup-policy.json`, then the
  ongoing controller only needs `controller-policy.json`.
- **`PassRole` is scoped.** The controller may pass only `spot-train-role`, and
  only to `ec2.amazonaws.com` — a compromised controller can't hand out arbitrary
  roles.
- **`ssm:StartSession`** in the controller policy is a human-operator convenience
  (to `tail -f` the box live). It's not needed by an automated cloud controller —
  drop that statement there. The box side (`AmazonSSMManagedInstanceCore` on the
  worker role) is attached by `setup`.

## For a throwaway test user

Attach **both** `setup-policy.json` and `controller-policy.json` (it does setup
*and* runs). Quick-and-dirty alternative on a disposable personal account:
`AmazonEC2FullAccess` + `AmazonS3FullAccess` + `IAMFullAccess` +
`AmazonSSMReadOnlyAccess` + `ServiceQuotasReadOnlyAccess`.
