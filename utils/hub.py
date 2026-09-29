"""Local-first model loading: a model already in the cache never touches the network.

huggingface_hub, diffusers and transformers check the Hub for a newer revision before they
use the cache. Offline, huggingface_hub 0.36 retries each of those requests 5 times with
backoff (1+2+4+8+8 s = 23 s per file): measured on the sd-turbo + TinyVAE + depth
ControlNet set, 62 failed connections and ~4 minutes added to start-up, everything then
loading from the cache anyway.

local_first(load, *args, **kwargs) calls load(..., local_files_only=True) first and goes to
the Hub only when the files are missing from the cache. Trade-off: a newer revision of an
already cached model is no longer picked up automatically (delete it from the cache, or
load it once with local_files_only=False, to update it).
"""
import logging


def _describe(args, kwargs) -> str:
    for value in (kwargs.get("repo_id"), kwargs.get("pretrained_model_name_or_path"),
                  args[0] if args and isinstance(args[0], str) else None,
                  args[1] if len(args) > 1 and isinstance(args[1], str) else None):
        if value:
            name = str(value)
            if kwargs.get("filename"):
                name += f"/{kwargs['filename']}"
            elif kwargs.get("subfolder"):
                name += f"/{kwargs['subfolder']}"
            return name
    return "model"


def local_first(load, *args, **kwargs):
    """load(*args, **kwargs) from the local cache, downloading only what is missing.

    load is any from_pretrained / from_single_file / hf_hub_download / load_lora_weights /
    load_ip_adapter, or a wrapper forwarding its kwargs to one. A caller that already set
    local_files_only keeps its choice."""
    if "local_files_only" in kwargs:
        return load(*args, **kwargs)
    try:
        return load(*args, local_files_only=True, **kwargs)
    except (OSError, ValueError) as e:
        # Not (fully) in the cache: huggingface_hub raises LocalEntryNotFoundError (an OSError
        # and a ValueError), diffusers / transformers an OSError. Anything else propagates.
        logging.info(f"[Hub] {_describe(args, kwargs)} not in the local cache, downloading "
                     f"({type(e).__name__})")
        return load(*args, **kwargs)
