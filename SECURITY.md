# Security policy

## Reporting a vulnerability

Please report vulnerabilities through GitHub's private security-advisory
feature rather than a public issue. Include the affected revision, impact, and
the smallest reproduction you can safely provide. Do not include credentials,
private genomic data, or other sensitive records in an issue or pull request.

## Credentials and local data

This repository does not require checked-in credentials. Supply optional
service credentials through environment variables and keep `.env` files,
cloud configuration, private keys, downloaded genomes, checkpoints, and
experiment outputs outside source control.

If a real credential is committed, revoke or rotate it immediately. Removing
it in a later commit is not sufficient because it remains available in Git
history and existing clones.

Only public or appropriately licensed genomic data should be used with the
included data-preparation tools. Do not commit controlled-access, identifiable,
clinical, or otherwise sensitive genomic data.

## Supported versions

Security fixes are applied to the latest revision on the default branch. This
is research software and is not intended for clinical or diagnostic use.

| Version | Supported |
| --- | --- |
| Latest `main` revision | Yes |
| Older revisions and local forks | No |

Reports are handled on a best-effort basis; no response-time or remediation
service-level agreement is offered. Enable GitHub private vulnerability
reporting before making the repository public.
