# Vendored UI libraries

Third-party browser scripts the web UI loads. They are committed as-is, embedded into the
binary with `include_str!` in `src/web.rs` (one static route per file under `/assets/`),
and never fetched from a CDN at request time. Nothing here is built or transpiled by this
repository.

Versions below were read from each file's own banner or version constant. Where a file
carries none, it says "unknown" rather than a guess.

| File | Library | Version | Upstream | License (as stated in the file, else upstream) |
|---|---|---|---|---|
| `cytoscape.min.js` | Cytoscape.js | 3.28.1 (the `version` constant in the bundle; banner is "Copyright (c) 2016-2023") | <https://github.com/cytoscape/cytoscape.js> | MIT (banner) |
| `layout-base.js` | layout-base (Bilkent i-Vis) | unknown (webpack bundle, no banner) | <https://github.com/iVis-at-Bilkent/layout-base> | no license header at the top; the bundle embeds an Apache-2.0 text for the SVD/matrix code. Upstream declares MIT (not verified here) |
| `cose-base.js` | cose-base (Bilkent i-Vis) | unknown (webpack bundle, no banner) | <https://github.com/iVis-at-Bilkent/cose-base> | no license header; upstream declares MIT (not verified here) |
| `cytoscape-fcose.js` | cytoscape-fcose layout | unknown (webpack bundle, no banner) | <https://github.com/iVis-at-Bilkent/cytoscape.js-fcose> | no license header; upstream declares MIT (not verified here) |
| `marked.min.js` | marked | 12.0.0 (banner) | <https://github.com/markedjs/marked> | MIT (banner) |
| `dompurify.min.js` | DOMPurify | 3.4.13 (banner and `version` constant) | <https://github.com/cure53/DOMPurify> | Apache-2.0 or MPL-2.0 (banner) |

Dependency order matters: the fcose layout needs `cose-base`, which needs `layout-base`,
and the fcose plugin registers itself against Cytoscape. `src/web.rs` serves them as
separate files and `assets/ui/index.html` loads them in that order.

## Updating a library

1. Download the release you want from the upstream repository or its npm package. The
   minified browser (UMD) bundle is the file to copy: for example `dist/cytoscape.min.js`
   from `cytoscape`, `dist/purify.min.js` from `dompurify` (saved here as
   `dompurify.min.js`), the UMD build of `marked`, and the webpack bundles
   `layout-base.js`, `cose-base.js` and `cytoscape-fcose.js` from their packages. The
   Bilkent libraries have to be taken as a compatible set: check each one's declared
   dependency range against the others before bumping one.
2. Replace the file in place, keeping the filename, so the `include_str!` path in
   `src/web.rs` does not change. A renamed file also needs that route updated.
3. Update the version, upstream and license columns above from the new file's banner. If the
   new bundle carries no banner, keep "unknown" and note where the file came from instead.
4. Smoke-test the UI against a live server: the document view (markdown rendering goes
   through `marked` then `DOMPurify`), the full-KB and per-document graph views (fcose
   layout), and the editor. A DOMPurify or marked bump is a security-relevant
   change; read the upstream release notes first.
5. Record the change under `[Unreleased]` in `CHANGELOG.md`.

Dependabot does not track these files (they are not an npm or Cargo dependency of this
repository), so a new upstream security release has to be noticed by hand. DOMPurify is
the one to watch.
