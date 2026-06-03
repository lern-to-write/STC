# Changelog

## Refactor: standalone `stc` package

Compared with the original ReKV-bundled STC code:

* `stc/cache/` was renamed to `stc/cacher/`.
* `STC_CACHE` is now `STCCache`, a regular per-stream state object. A legacy process-wide default remains available through `stc.default_cache()`.
* `STC_Pruner` is now `STCPruner`.
* CLIP and SigLIP patch helpers were merged into `stc.integrations.hf_vit.register_stc_cacher(..., kind=...)`.
* Pruner internals were split into `anchors`, `scoring`, `index_mapper`, and `specs`.
* ReKV STC controls now use environment variables instead of STC-specific CLI arguments.
