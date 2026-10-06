# Troubleshooting

Common issues and fixes for mcp-md-wiki.

## Backup and Recovery

Of the three stateful things this project touches — the KB's git clone, `state.db`,
and the Qdrant collection — **only the git repo is authoritative.** `state.db` (the
SQLite state and metadata index) and Qdrant's vectors are both fully **derived** from
the corpus: every row and every point can be rebuilt from the markdown files and their
frontmatter via `mcp-md-wiki index --full`. So the short version of a backup policy
here is: back up the git repo the way you'd back up any other git-hosted content
(forge-level backups, a mirror remote, or trusting the forge's own durability), and
you've covered the one thing that genuinely cannot be reconstructed. Backing up
`state.db` or the `qdrant` volume is optional — see "What backing up the derived
stores buys you" below for what it saves you over a rebuild.

### Recovering after losing `state.db` and/or the Qdrant volume

1. Confirm (or restore) that the KB clone is intact and reachable at `GIT_URL`/
   `GIT_BRANCH`. If the container's data volume was wiped entirely, a fresh start
   re-clones it automatically — see [Knowledge Base Storage](USAGE.md#knowledge-base-storage)
   in USAGE.md.
2. Run a full reindex:

   ```bash
   docker compose exec mcp-md-wiki mcp-md-wiki index --full
   ```

   This drops and recreates the Qdrant collection, clears `state.db`'s
   `indexed_files`/`documents`/`document_fields` tables, and re-processes every file
   from scratch — re-chunking, re-embedding, and rebuilding the metadata index.
3. Confirm the recovery with `mcp-md-wiki status --json` (or `GET /status`): a healthy
   result has `qdrant_points_deficit` at or near zero, and `indexed_files`/`documents`
   counts matching your corpus size.

### `index --full` refuses to run: an invalid schema blocks recovery

If any `.kb-schema.yaml` in the tree is currently invalid, `index` (with or without
`--full`) **refuses to run at all**, before anything is dropped or cleared. The error
starts `Refusing to index while a .kb-schema.yaml is invalid` and lists every invalid
file with its reason, one `- <path>: <reason>` line each.

So a disaster recovery can be blocked by a pre-existing, unrelated schema problem —
see [The server refuses to start, or stops indexing, over a schema file](#the-server-refuses-to-start-or-stops-indexing-over-a-schema-file)
below for how to fix it, then retry `index --full`.

### A Qdrant wipe with `state.db` intact

If only the Qdrant volume is lost — a `docker volume rm` on the wrong volume, disk
corruption isolated to Qdrant's storage — while `state.db` survives, the failure mode
is different from a full loss: the server's startup `ensure_collection` step silently
recreates an empty collection, and because `state.db` still believes every file is
already indexed and unchanged, an ordinary incremental reindex or reconcile sweep
finds nothing to do. Historically (#155) this was a **silent, permanent blackout** —
search would run against an empty collection with no error surfaced anywhere.

As of the current server, this is detected passively: `GET /status` and
`GET /metrics` compare `state.db`'s summed chunk count against Qdrant's actual point
count on every request, and report an error (in `store.errors`, and via the
`kb_qdrant_points_deficit` Prometheus gauge) whenever Qdrant is short by more than a
small slack — a real wipe blows past that slack by orders of magnitude, since it
zeroes the whole collection at once rather than losing a handful of points. Detection
is currently passive only — the check tells you something is wrong, it does not
itself trigger a reindex. If you see this error, or a `kb_qdrant_points_deficit`
metric that stays large and positive, run `mcp-md-wiki index --full` to force a full
reconcile (subject to the invalid-schema caveat above).

### What backing up the derived stores buys you

Skipping a backup of `state.db`/the Qdrant volume is safe — nothing about document
*content* is ever at risk, since that only ever lived in the git repo — but a rebuild
from scratch isn't free:

- **Time, and on a paid embedding endpoint, money.** A full reindex re-embeds every
  chunk in the corpus from nothing; there's no partial-credit path. For a large KB, or
  a metered embedding API, that's a real cost worth weighing against the effort of
  also backing up the Qdrant volume and `state.db`.
- **`indexed_at` history.** Rebuilt rows get a fresh `indexed_at`, so anything that
  trends "when was this document last touched" loses that history across the rebuild.
- **In-flight `/status` counters.** The last completed run's outcome tallies and any
  run-in-progress state are process-global and reset on restart regardless of what's
  backed up — they aren't part of either persisted store to begin with.

## Startup Failures

### `unknown field '...'` on startup

Your `config.yaml` has fields that no longer exist (e.g. `chunking.strategy`, `chunking.chunk_overlap`, `webhook.port`). The schema uses `#[serde(deny_unknown_fields)]`, so any unrecognized key is a hard error.

**Fix:** Compare your config against [config.example.yaml](config.example.yaml) and remove any fields not present in the example.

### `Missing required environment variable(s)` (EMBEDDING_BASE_URL, EMBEDDING_MODEL, QDRANT_URL)

Running outside Docker without the required environment variables set. The error lists all missing vars at once, by bare env var name:

```text
Missing required environment variable(s):
  - EMBEDDING_BASE_URL
  - EMBEDDING_MODEL
  - QDRANT_URL
```

**Fix:** Either run via `docker compose` (which wires env vars automatically) or export them manually:

```bash
export EMBEDDING_BASE_URL=http://localhost:8080/v1
export EMBEDDING_MODEL=nomic-embed-text-v2-moe
export QDRANT_URL=http://localhost:6334
```

### `MCP_BEARER_TOKEN` not set

The server refuses to start with no authentication at all: OAuth is not enabled and no static bearer token is set (unless `mcp.allow_unauthenticated: true` is set in config).

**Fix:** Enable `mcp.oauth` (recommended, see [docs/oauth.md](../docs/oauth.md)), or set `MCP_BEARER_TOKEN` in your `.env` file.

### `mcp.oauth.enabled is true but the OAuth config is not usable`

The error lists every problem at once, one per line. The common ones:

- an empty `issuer`/`resource`/`audience`;
- an `issuer`/`resource`/`jwks_uri` that is not an absolute http(s) URL;
- `HS256` or `none` in `algorithms` (never allowed);
- a blank `required_scope` (`required_scope: ""`) or one containing a space;
- a `required_scopes` entry that is blank (`mcp.oauth.required_scopes contains
  an empty entry`) or contains a space (`mcp.oauth.required_scopes entry "a b"
  must be a single scope`) — list each scope as its own entry;
- `leeway_secs` above 300.

**Fix:** Correct the named keys. [docs/oauth.md](../docs/oauth.md#configuration) has the full table.

### `destination path '.' already exists and is not an empty directory`

The auto-clone on startup found a non-empty `DATA_PATH` without a `.git` directory. This happens when files already exist in the volume (e.g. leftover data from a previous setup).

**Fix:** Remove the volume and let the auto-clone start fresh:

```bash
docker compose down mcp-md-wiki
docker volume rm <project>_kb_data
docker compose up -d mcp-md-wiki
```

### `/data/.git: Permission denied` on startup

The image runs as `nonroot` (uid 65532) unconditionally — this is baked into the Dockerfile, not something you opt into with a compose `user:` directive. Docker initializes a fresh named volume mount point by copying ownership from the image's directory at that path, if one exists. Images built before this was fixed had no `/data` directory in the image, so Docker created the mount point `root:root`, and the non-root container couldn't write to it — hitting every fresh deployment, `user:` uncommented or not.

Images from this fix forward create and chown `/data` in the image, so a fresh named volume comes up already writable by `nonroot` and this shouldn't occur. If you hit it anyway (e.g. a volume created by an older image before upgrading), fix the existing volume's ownership once:

```bash
docker run --rm -v <volume_name>:/data --user root --entrypoint chown \
  ghcr.io/st0nefish/mcp-md-wiki:latest 65532:65532 /data
```

If you're running the container with a custom compose `user:` override instead of the image default, replace `65532:65532` with that uid:gid. This only needs to be done once per pre-existing volume.

`:latest` is the most recent release; use `:dev` (newest `master` build) or `:vX.Y.Z` if that is the tag your stack runs.

### `docker compose up` hangs; `mcp-md-wiki` never starts; `qdrant` shows `starting` then `unhealthy`

**Symptom.** `docker compose ps` shows `qdrant` stuck in `(health: starting)` and then
`(unhealthy)` once its `retries` are exhausted. `mcp-md-wiki` never even shows as `Created`
starting a container — `depends_on: qdrant: condition: service_healthy` means compose
won't start it until qdrant reports healthy, which here it never will. This hits a
clean `docker compose up` on any host: a first-time deployment, a disaster-recovery
rebuild, or a reboot. If the stack looks fine on your existing host, that's because
Watchtower restarts the `qdrant` container directly when it pulls a new image, bypassing
compose and its `depends_on` gate entirely — it doesn't mean the healthcheck works.

**Cause.** Fixed as of #256: the qdrant healthcheck used to run `curl -f
http://localhost:6333/health`, and both halves were wrong — the `qdrant/qdrant` image
ships no `curl` (or `wget`), so the command can't execute at all, and even if it could,
current Qdrant dropped `/health` in favor of `/readyz`/`/healthz`/`/livez`. The
healthcheck could never report anything but `unhealthy`.

**Fix.** If you're on a version of this repo before #256, `git pull` or update your
compose file — `docker-compose.yml` and every template under `deploy/templates/` now
probe the gRPC port directly with a `/dev/tcp` bash redirect instead of curl:

```yaml
healthcheck:
  test: ["CMD", "bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/6334"]
```

If you're maintaining a compose file that forked from an older version of this repo,
apply the same change by hand. Confirm it with `docker inspect --format
'{{json .State.Health}}' <qdrant container>` — `"Status":"healthy"` after the
`start_period` elapses.

## Embedding Service

### Container exits immediately

Usually means the Docker image doesn't match your hardware. The CPU image (`server`) works everywhere but is slower. GPU images need matching drivers.

**Fix:** Check the compose templates in `deploy/templates/` or the [Embedding Backends](../README.md#embedding-backends) section in the README and pick the right image for your hardware.

### `model file not found`

The `MODEL_PATH` or `MODEL_FILE` in `.env` doesn't match where the model is on disk.

**Fix:** Verify the model file exists at `$MODEL_PATH/$MODEL_FILE` and the path is correct in `.env`.

### Slow embeddings

You're likely running the CPU image when a GPU is available.

**Fix:** Switch to the appropriate GPU image in `docker-compose.yml` (see commented blocks) and add `-ngl 999` to offload all layers to GPU.

On AMD RDNA3/RDNA4, check `rocm-smi` at idle afterwards — offloading an encoder
through ROCm can pin the card at 100% forever. See [GPU Backends](#gpu-backends).

## GPU Backends

### GPU pinned at 100% while completely idle (AMD ROCm)

**Symptom.** An AMD GPU serving embeddings or reranking sits at 100% utilisation,
core clock at maximum, and 80–100W draw — continuously, while the server handles
no requests at all. Memory clock stays at idle, which is the tell: real inference
saturates memory bandwidth, a busy-wait loop does not.

Confirm it with `rocm-smi` while nothing is being indexed or searched:

```text
GPU  Temp    AvgPwr  SCLK     MCLK   VRAM%  GPU%
0    68.0c   97.0W   3423Mhz  96Mhz  76%    100%    <- spinning
1    38.0c   14.0W   41Mhz    96Mhz  78%    3%      <- healthy idle
```

Stop the container; if utilisation drops to ~3% and power to ~15W, that container
is the cause.

**Cause.** An AMD MES firmware bug affecting RDNA3/RDNA4 (gfx11xx/gfx12xx), tracked
as [ROCm/ROCm#5706](https://github.com/ROCm/ROCm/issues/5706). It is a HIP runtime
problem, not a llama.cpp one — it also reproduces under PyTorch-ROCm with no
workload at all ([ROCm/ROCm#6298](https://github.com/ROCm/ROCm/issues/6298)).

On this project's workload it appears specifically when an **encoder** model is
offloaded to GPU (`-ngl` > 0): both embedding and reranking servers trigger it, in
either `--embeddings` or `--reranking` mode. Decoder models on the same image and
host do not.

**Fix, in order of preference:**

1. **Update the GPU firmware.** MES firmware `0x8b` (amdgpu 31.20.0+) resolves it at
   the source. Check what you have:

   ```bash
   sudo grep -i "^MES feature" /sys/kernel/debug/dri/*/amdgpu_firmware_info
   ```

   `0x81` and similar are affected. Note that a distribution `linux-firmware`
   package may be far older than your GPU — check its snapshot date before
   assuming an upgrade will help.

2. **Use the Vulkan backend for the encoders** (`compose-vulkan.yml`). Vulkan does
   not go through HIP and does not exhibit the bug. It costs roughly 40–50% more
   time per rerank batch than ROCm, which is usually a good trade against ~80W
   burned continuously. GGUF models and all server flags carry over unchanged —
   only the image tag differs.

3. **Run the encoders on CPU.** Avoids the spin, but for reranking specifically
   expect roughly 25–30× the latency; measured at 76s versus 2.8s for 150
   candidates on one deployment. Viable for embeddings, which are cheap per query;
   generally not viable for reranking.

Things that do **not** fix it, all tested: `--parallel 1`, `--poll 0`,
`HSA_ENABLE_INTERRUPT=1`, removing `HSA_OVERRIDE_GFX_VERSION`, and
`GPU_MAX_HW_QUEUES=1` (which is reported to work for decoder models, but does not
help encoders).

### Reranking returns 500 on every request

**Symptom.** `Reranker unavailable, falling back to fused order` in the logs, or
searches that take tens of seconds and then return unranked results. Querying the
reranker directly returns:

```json
{"error":{"code":500,"message":"input (518 tokens) is too large to process. increase the physical batch size (current batch size: 512)"}}
```

**Cause.** A single candidate document exceeds the reranker's physical batch size, and
the server rejects the whole request rather than that one document. llama.cpp
defaults `--ubatch-size` to 512 tokens; documents at the default
`reranking.max_document_bytes` of 6144 bytes (sized for an 8192-token reranker served
with `--ubatch-size` equal to `--ctx-size`) can exceed that on a smaller-context
server.

**Fix.** mcp-md-wiki truncates each document to `reranking.max_document_bytes` bytes
before sending it, so an over-long document degrades *that document's* score instead
of costing the whole query its reranking. When this happens you get one warning per
request rather than a silent fallback:

```text
Truncated 1 of 50 rerank documents to the 6144-byte reranking.max_document_bytes budget (longest was 7200 bytes).
```

Lower `reranking.max_document_bytes` to fit your reranker's actual `--ubatch-size`, or
raise `--ubatch-size`/`--ctx-size` to fit the configured budget — whichever is cheaper
for your deployment. Unlike `chunking.max_chunk_size`, this key has no automatic
reindex tie-in and no other consumer, so changing it never re-chunks or re-embeds
anything; it only changes what the reranker is sent (restart required, since
`RerankClient` bakes it in at construction). It is applied only to the reranker's view
of the text; returned document content is unaffected.

That bound is in **bytes**, not tokens, and the two are not the same thing. Roughly
four bytes per token holds for English prose (6144 bytes ≈ 1500 tokens), but CJK
text, base64 blobs and long unbroken identifiers tokenize far worse per byte. The
default is sized for the worst case — a byte-level BPE tokenizer never emits more
than one token per byte, so 6144 bytes plus 2048 tokens of query headroom always
fits an 8192-token `--ubatch-size`. When you size the key down for a smaller
`--ubatch-size`, the English-prose estimate is optimistic: `1500` suits a
512-token ubatch for mostly-English prose, but a corpus that isn't needs a lower
budget, or the reranker's batch sizing raised to match its context:

```yaml
      --ctx-size 8192
      --batch-size 8192
      --ubatch-size 8192
```

Size `--ctx-size` for the number of parallel slots as well — llama.cpp divides it
across them, and it will cap per-slot context to the model's training context.
Going too large fails at startup with `failed to fit params to free device memory`
when the card is shared with another model.

**Note on the "tens of seconds" symptom.** A rerank failure is now given a ~5s retry
budget rather than 120s. A 500 like the one above is deterministic — it never succeeds
on retry — so the old budget turned a sub-second failure into an MCP tool timeout that
made search look broken. Reranking is an enhancement whose fallback (fused order) is
serviceable, so it now gives up quickly and returns results.

### Reranking is enabled but never runs

**Symptom.** `reranking.enabled: true`, the reranker container is healthy, but
`search` with `explain: true` returns `pre_rerank_score: null` on every result and
ranking looks like plain vector fusion.

**Cause.** The reranker is being called with an empty document list, so it returns
immediately and the fused order stands. Historically this was caused by a payload
key mismatch between what indexing wrote and what retrieval read (fixed in #125,
which routes both through `qdrant::CHUNK_TEXT_KEY`).

**Fix.** Confirm the reranker is reachable and returns scores when called directly.
If it does, and `pre_rerank_score` is still null, the candidate list is empty before
it is sent — check that chunks carry text in the payload field retrieval reads.

## Indexing

### Files skipped: "missing required frontmatter field"

Files are missing a field listed in `frontmatter.required`.

**Fix:** Either add the missing field to the file's frontmatter, or remove it from the `required` list in your config. Run `mcp-md-wiki validate` to check all files.

### `target_chunk_size must be <= max_chunk_size`

The values are swapped — `target_chunk_size` must be the smaller value.

**Fix:** Swap the values in your config. Example: `target_chunk_size: 1000`, `max_chunk_size: 1500`.

### The server refuses to start, or stops indexing, over a schema file

**Symptom.** One of:

- The server exits at startup with `Refusing to start: every .kb-schema.yaml in the
  indexed tree must be valid`, followed by the list of invalid files.
- A running server keeps working, but nothing new gets indexed: `GET /status` has a
  `schema_error` entry, `kb_schema_invalid` is 1 in `/metrics`, and the log repeats
  `schema rebuild REFUSED, keeping the previous schema` on every reconcile sweep, and
  `Indexing run REFUSED: a .kb-schema.yaml is invalid` after a write. Writes still
  succeed and validate against the previous schema.
- Moving a directory fails with `the schema file '<path>' inside it is invalid`.
- `mcp-md-wiki validate` or `mcp-md-wiki index` fails with `SCHEMA ERRORS`.

**Cause.** A `.kb-schema.yaml` in the indexed tree that is present must be valid, and
an invalid one is never loaded: at startup that is fatal, and on a running server (the
file arrived in a push) the rebuild is refused and the last schema that loaded cleanly
stays in effect — which is also why a directory move carrying the broken file is
refused. One in a hidden directory, or in one `indexing.exclude` rules out entirely, is
never read. A file is invalid when it is malformed YAML, uses an
unknown key (including under `dedup:`), contradicts itself (an unrecognized
`$`-prefixed placeholder in a `values:` list, a field declaring both a scalar `type`
and nested `fields:`, one path declared twice, a `dedup.threshold` outside 0.0–1.0), or
is over 256 KB — refused on size alone, never read or parsed.

**Diagnosis.** Run:

```bash
mcp-md-wiki validate
```

Every invalid schema file is listed, with the parse or validation error:

```text
SCHEMA ERRORS (1):
  food/recipes/.kb-schema.yaml: field 'planning' declares type 'text' but also nested fields; a field is either a value or a container, not both
```

On a running server, `GET /status`'s `schema_error.files` carries the same list.

**Fix.** Correct (or revert) the named `.kb-schema.yaml` in the knowledge base's git
repository and push. `update_schema` can't edit a file that no longer parses, and the
document tools refuse schema-file paths. See
[Directory Schemas](USAGE.md#directory-schemas-kb-schemayaml) in USAGE.md for the
authoring rules. On a running server the push's webhook queues a full reconcile: the
rebuild succeeds, `schema_error` clears, and every document written or changed while
the schema was refused is indexed. A server that would not start just needs starting
again once the file is fixed.

### Qdrant connection refused

Qdrant isn't running or hasn't finished starting.

**Fix:** Check `docker compose ps` — wait for the Qdrant healthcheck to pass. The default healthcheck has a 10-second start period. If running outside Docker, verify the URL in `QDRANT_URL` points to the gRPC port (6334, not 6333).

## Webhook

### 404 on `/hooks/reindex`

The webhook route is only mounted when `WEBHOOK_SECRET` is set to a non-empty value.

**Fix:** Set `WEBHOOK_SECRET` in your `.env` and restart the service.

### 401 Unauthorized

The HMAC secret in your Git forge doesn't match `WEBHOOK_SECRET`.

**Fix:** Ensure the secret value is identical in both your `.env` and your Git forge's webhook settings. Also check that `webhook.provider` matches your forge (`gitea`, `github`, or `gitlab`).

### 200 OK but no reindex happens

The push was to a branch that doesn't match `GIT_BRANCH`.

**Fix:** Check that the webhook fires on pushes to the branch configured in `GIT_BRANCH` (default: `master`).

### Git fetch failed (500 Internal Server Error)

The webhook verified successfully but the in-container `git fetch` or `git merge` failed. Common causes:

- **Wrong `GIT_URL`** — check the URL is correct and reachable from inside the container.
- **Bad or expired `GIT_PULL_TOKEN`** — for private HTTPS repos, the token needs read repository access for webhook pulls (and **write** access if you use the MCP write tools, which push commits back). Regenerate it in your forge's settings.
- **SSH URL without keys** — SSH URLs bypass token injection, but the container needs SSH keys configured. For Docker deployments, HTTPS with a token is simpler.
- **Diverged history** — the merge uses `--ff-only` and will fail if the local branch has diverged. This usually means someone modified files directly in the bind-mounted directory. Using a named volume (recommended) avoids this by keeping the repo inaccessible from the host.

**Fix:** Check `docker logs mcp-md-wiki` for the specific error (tokens are redacted in log output). Verify the URL and token work from the host:

```bash
git ls-remote "https://<token>@your-forge.example.com/org/repo.git"
```

### 302 redirect on webhook

Your reverse proxy is redirecting HTTP to HTTPS, but the webhook is configured with an HTTP URL.

**Fix:** Update the webhook URL in your forge to use `https://`.

## Access and Networking

### `curl` to `/health` or `/status` returns a 302 redirect to an SSO login page

This is the default proxy-with-SSO posture doing exactly what it's configured to do — see [Deployment Posture: Network Exposure and Access Control](USAGE.md#deployment-posture-network-exposure-and-access-control) in USAGE.md for the full picture. The reverse proxy in front of the container (Authentik via Traefik, in the reference deployment) intercepts the request before it ever reaches mcp-md-wiki and redirects it to the identity provider's login page, because the proxy has no way to distinguish a diagnostic `curl` from a browser that can complete the SSO flow. The 302 never originates from the binary itself: `/health` handles GET with no gate at all, and `/status` is bearer-gated, not SSO-gated — neither returns a redirect on its own.

Passing a bearer token doesn't change the outcome. This was confirmed against the reference deployment during verification: `curl` with `-H "Authorization: Bearer $MCP_BEARER_TOKEN"` and `curl` with no header at all returned the identical 302 to Authentik's login page, because the token is meaningless to the proxy's SSO layer — it redirects unauthenticated *browsers*, it does not inspect bearer credentials.

**Two ways out, neither requiring an interactive SSO session from a script:**

1. **Reach the container directly, bypassing the proxy.** From inside the Docker network: `docker compose exec mcp-md-wiki curl http://localhost:8001/health`. From the host, if the port happens to be published: `curl http://localhost:8001/health` against the mapped port. Either way skips the proxy hop entirely, so nothing intercepts the request or has a chance to redirect it.
2. **Use the bearer token against a route the proxy doesn't intercept.** `/status` is gated inside the binary itself (the same [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server) crate's `require_auth` layer that protects `/mcp`, mcp-md-wiki#308), which is a different mechanism from the proxy's SSO layer that's producing the 302:

   ```bash
   curl -H "Authorization: Bearer $MCP_BEARER_TOKEN" https://your-host/status
   ```

   This only helps if your proxy is configured to pass `/status` through without intercepting it — if the proxy blankets every path under SSO regardless of route, it will 302 this request too, and option 1 (reach the container directly) is the only way to get a diagnostic read.

If you want `/health` and `/status` reachable from the LAN without a proxy hop or an SSO session at all, that's exactly what the LAN-only posture in [Deployment Posture: Network Exposure and Access Control](USAGE.md#deployment-posture-network-exposure-and-access-control) documents — publishing the port to a specific interface trades the SSO gate for LAN reachability as the access control, with the write-surface trade-off spelled out there.

## OAuth

Start with the startup log. `OAuth: authorization server signing keys loaded` means
the issuer and JWKS are fine. `OAuth: could not load the authorization server's
signing keys` carries the reason. The usual one is `metadata issuer … does not
match mcp.oauth.issuer`: the configured issuer differs from the provider's by a
trailing slash or a path. Copy it exactly from the provider's discovery document.

When a request is refused, the `OAuth bearer auth rejected` warning gives the
reason. To see what the token actually carries, decode its middle segment
(base64url JSON).

| Reason in the log | Meaning and fix |
|---|---|
| `credential is not a JWT` | The token is opaque: Authelia's default, and some other providers'. Configure the provider to issue JWT access tokens (Authelia: `access_token_signed_response_alg: 'RS256'` on the client). Also seen when a mistyped static token is sent while OAuth is on. |
| `token rejected: InvalidAudience` | The token's `aud` is not in `audience`/`audiences`. Put the value your provider actually stamps there: the resource URL or the client_id. See [Choosing the audience](../docs/oauth.md#choosing-the-audience). |
| `token rejected: InvalidIssuer` / `token iss is not a single string…` | `issuer` does not match the token's `iss` byte for byte. Check for a trailing slash. |
| `token rejected: ExpiredSignature` / `ImmatureSignature` | The token has expired, or it is not valid yet (`nbf`). Check clock sync on both hosts. `leeway_secs` absorbs small drift. |
| `token algorithm … is not in mcp.oauth.algorithms` | The provider signs with an algorithm you excluded, or with HS256. HS256 cannot be verified by a resource server; give the provider an asymmetric signing key. |
| `token typ … is not accepted` | The header `typ` is `JWT` or missing while `require_at_jwt` is on, or it is a different kind of JWT altogether. |
| `token header lists critical extensions (crit)…` | The provider marked the token with a JWS extension this server cannot process. Configure it not to. |
| `token nbf is not a NumericDate…` | The provider emitted a malformed `nbf` (a string, or an out-of-range number). |
| `token is sender-constrained (cnf)…` | The token is DPoP- or mTLS-bound, which this server cannot verify; it accepts plain bearer tokens only. Configure the client to request unbound tokens. |
| `no … key for kid …` | No key in the JWKS matches. Either the provider rotated its keys (the next refetch, at most a minute later, picks the new one up), or the token comes from a different issuer or client. Kanidm, for example, has a separate issuer and JWKS per client. |
| `InsufficientScope` (403) | The token is valid but lacks at least one required scope — every scope in `required_scope` ∪ `required_scopes` (by default just `mcp:read`) must be present. The info line `OAuth token is valid but lacks the required scope` lists the scopes it has (`present=[…]`) next to the ones required (`required=…`). Grant the missing scope(s) to the user or client at the provider, and make sure `scopes_supported` lists every required scope — clients request exactly what it advertises. If the scopes are in a claim other than `scope`/`scp`, add that claim to `scope_claims`. |

## Model / Vector Issues

### Qdrant dimension mismatch after model change

Changing the embedding model (or `EMBEDDING_VECTOR_SIZE`) makes existing vectors incompatible.

**Fix:** Run `mcp-md-wiki index --full` to drop and recreate the Qdrant collection with the new dimensions.

## Logging and Observability

### `RUST_LOG` syntax and useful presets

Set `RUST_LOG` in your `.env` file or compose environment. The server warns on stderr if the value is unparseable and falls back to `info`.

| Value | Effect |
|---|---|
| `info` | Default — startup events, webhook accepts, indexing summaries |
| `mcp_md_wiki=debug` | Verbose app logging: per-file decisions, search timing, per-batch embed progress |
| `info,mcp_md_wiki::webhook=debug` | Info everywhere + detailed webhook trace |
| `info,mcp_md_wiki=debug,oauth_resource_server=debug` | App debug logging plus auth debug lines (see below) — the `mcp_md_wiki=debug` preset alone does NOT show them, since auth now logs under a different target |
| `debug` | Very verbose — includes library internals (noisy) |

### Key log events and how to find them

Authentication events (mcp-md-wiki#308) are logged under the `oauth_resource_server` target, not `mcp_md_wiki` — a `RUST_LOG` that names only `mcp_md_wiki` (e.g. `mcp_md_wiki=debug`) will not show them; add `oauth_resource_server=debug` as in the preset above.

`Bearer auth rejected` (WARN) — every request rejected by MCP bearer-token auth when OAuth is off. A flood of these means a client is using the wrong token.

`OAuth bearer auth rejected` (WARN) — a request refused with OAuth on, with the reason (`reason=Invalid("…")` or `InsufficientScope`). See [OAuth](#oauth) for what each reason means. A request with no credential at all is logged only at DEBUG (`No bearer credential presented`) under the `oauth_resource_server` target, because every OAuth client's first request looks like that.

`Webhook signature verification failed` (WARN) — HMAC mismatch between the forge secret and `WEBHOOK_SECRET`, logged with the provider name.

`Webhook pull applied; marking changed paths dirty` (INFO, logged with `provider`, `branch`, and a `changed` count) — the webhook passed signature + branch checks, fetched and merged the pull, diffed the range it pulled in, and marked exactly those paths dirty for the reindex worker. `Webhook accepted with no git_url configured; marking a full reconcile` (INFO) is the same acceptance path when `source.git_url` is unset: there's nothing to fetch or diff, so the whole corpus is marked for a full reconcile instead of specific paths.

Either way, marking is instant — the response (`200 "Changes queued for indexing"`) returns before any actual indexing work happens. There is no "reindex already in progress, coalescing/skipping" log line anymore, because there is no longer a single-flight lock a webhook can lose a race for: every accepted delivery's paths land on the same `ReindexQueue`, which coalesces overlapping work rather than dropping any of it. A burst of pushes shows as a run of `Webhook pull applied` lines, one per delivery, each with its own `changed` count — none of them skipped. To confirm the marked work actually got indexed (as opposed to just accepted), look for the reindex worker's own output: the `Indexing run complete` summary below, or a `warn!` naming a unit that's being retried or dropped after repeated failure.

`Indexing run complete` (INFO) — a structured per-run summary logged after every indexing run:

```text
Indexing run complete discovered=42 indexed=5 skipped=36 invalid=0 empty=0 read_errors=0 orphans_removed=1 elapsed_secs=3.2
```

| Field | Meaning |
|---|---|
| `discovered` | Total files matched by include/exclude patterns |
| `indexed` | Files successfully embedded and upserted |
| `skipped` | Unchanged files (hash matched state DB) — normal in incremental mode |
| `invalid` | Files that failed frontmatter validation |
| `empty` | Files that produced no chunks (blank body after frontmatter) |
| `read_errors` | Files that could not be read (permissions, encoding errors) |
| `orphans_removed` | Qdrant points removed for files no longer on disk |
| `elapsed_secs` | Wall-clock time for the run |

If a file you expected to be indexed shows up under `skipped`, its hash matches the last indexed version — it hasn't changed since the last run. If it shows under `invalid`, run `mcp-md-wiki validate` for details.

`git fetch timed out after Xs` / `git merge timed out after Xs` (ERROR) — webhook-triggered git subprocess exceeded the 120-second timeout. Check that `GIT_URL` is reachable from inside the container.

`Could not read mtime for '...', defaulting to 0` (WARN) — filesystem metadata was unavailable. The file is still indexed; `mtime` in the Qdrant payload will be `0`.

## Apple Silicon / macOS

### No Metal support in Docker

The llama.cpp Docker images don't support Apple Metal GPU acceleration. Docker on macOS runs a Linux VM, which doesn't have access to the Metal API.

**Options:**

1. **Run llama-server natively** — `brew install llama.cpp`, then run `llama-server` with your model and point `EMBEDDING_BASE_URL` at it (e.g. `http://host.docker.internal:8080/v1` if mcp-md-wiki is still in Docker).
2. **Use the CPU Docker image** — works but is slower than native Metal.
3. **Use an external API** — point `EMBEDDING_BASE_URL` at OpenAI, Ollama, or any OpenAI-compatible endpoint and remove the `embeddings` service from compose.
