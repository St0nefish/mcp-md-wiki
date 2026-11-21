// The former single `mod tests` of `src/mcp.rs`, split by tool family. Every
// test body is verbatim; each file imports the module it tests plus
// `crate::mcp`'s own items.
mod access;
mod get_document;
mod overlay;
mod payload;
mod schema;
mod search;
mod server;
mod write;
