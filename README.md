# pi-kilo

Kilo Code Gateway provider for [pi](https://github.com/earendil-works/pi), rebuilt for
fast startup. A lean rewrite of [Kilo-Org/kilo-pi-provider](https://github.com/Kilo-Org/kilo-pi-provider)
keeping the parts that matter: the model catalog, auth, and streaming.

## Why a rewrite

The original registers all 367 catalog models at startup. Every
`pi.registerProvider` call makes pi rebuild its whole model snapshot and run a
catalog refresh, and that cost scales with the catalog — 367 models added
~580ms to every single launch. pi-kilo registers the models you actually use:

- the **free tier** (12 models),
- models you **picked before** (sticky, persisted across sessions),
- anything matching `KILO_MODELS`.

That is ~12–15 models ≈ ~20ms. The full 367-model catalog still lives in the
disk cache and is one `KILO_MODELS=all` away.

## Install

```sh
pi install git:github.com/hakergeniusz/pi-kilo
```

## Login

```sh
/login kilo        # OAuth device flow, browser authorization
```

Credentials land in pi's `auth.json` under the same `kilo` entry the official
provider uses — an existing login carries over, no re-auth needed.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `KILO_API_KEY` | — | API key / token; else the OAuth credential is used |
| `KILO_ORG_ID` | — | Organization id; else the one stored at login |
| `KILO_API_URL` | `https://api.kilo.ai` | API base |
| `KILO_MODELS` | *(free + sticky)* | `all`, or comma-separated ids/globs (`gpt-5*`, `anthropic/claude-opus*`) |
| `PI_KILO_DEBUG` | — | `1` enables trace logging to stderr |

## Behavior notes

- **Startup never blocks on the network** once the catalog is cached (first
  run fetches it once). Stale entries serve immediately and revalidate in the
  background for the next launch.
- The catalog cache is shared with the original provider
  (`~/.pi/agent/cache/kilo-pi-provider/`), so switching between the two costs
  nothing.
- Models route through Kilo's OpenRouter-compatible gateway; OpenAI
  reasoning/frontier models use the Responses API with the gateway's
  `response.failed` quirk normalized.
- On first use of a Kilo model the Terms of Service notice is shown once.
