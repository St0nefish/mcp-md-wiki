//! Memory mode: the parts of the memory tool group that do not need a server — the
//! schema the binary supplies for the memory area ([`builtin_scopes`]) and the pure
//! rules the tools apply ([`logic`]). The tools themselves are `mcp/memory.rs`.

pub mod logic;

use std::path::PathBuf;

use crate::config::MemoryConfig;
use crate::schema::SchemaFile;

/// The memory area's schema, compiled in. Clients never see or author this
/// frontmatter; the tools write it.
const MEMORY_SCHEMA: &str = include_str!("../../assets/memory/schema.yaml");

/// The schemas the binary defines itself, by directory: the memory area's, while
/// `memory.enabled` is on, and nothing otherwise. Laid in beside the schema files on
/// disk by [`crate::schema::SchemaCache::build_with`], which refuses a file on disk
/// at or under such a directory.
pub fn builtin_scopes(config: &MemoryConfig) -> Vec<(PathBuf, SchemaFile)> {
    if !config.enabled {
        return Vec::new();
    }
    let Ok(root) = crate::policy::normalize_prefix(&config.root) else {
        // Refused at config load; nothing to lay in for a root that does not exist.
        return Vec::new();
    };
    match crate::schema::parse_schema_text(MEMORY_SCHEMA) {
        Ok(file) => vec![(PathBuf::from(root), file)],
        Err(e) => {
            // A compiled-in file; a unit test holds it valid.
            tracing::error!("the built-in memory schema is invalid: {e}");
            Vec::new()
        }
    }
}

/// The directory the memories live in, as the policy and path code write it
/// (`memory`, never `/memory/`).
pub fn root_dir(config: &MemoryConfig) -> String {
    crate::policy::normalize_prefix(&config.root).unwrap_or_else(|_| config.root.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn the_builtin_schema_parses_and_declares_what_the_tools_write() {
        let scopes = builtin_scopes(&on());
        assert_eq!(scopes.len(), 1);
        let (dir, file) = &scopes[0];
        assert_eq!(dir, &PathBuf::from("memory"));
        for field in [
            "id",
            "type",
            "status",
            "source_client",
            "source_url",
            "created",
            "updated",
            "supersedes",
            "confidence",
            "tags",
        ] {
            assert!(file.fields.contains_key(field), "{field} not declared");
        }
        let values = |f: &str| file.fields[f].values.clone().unwrap();
        assert_eq!(values("type"), logic::TYPES);
        assert_eq!(values("status"), logic::STATUSES);
    }

    #[test]
    fn nothing_is_built_in_while_the_group_is_off() {
        assert!(builtin_scopes(&MemoryConfig::default()).is_empty());
    }

    #[test]
    fn a_schema_file_on_disk_in_the_memory_area_makes_the_build_fail() {
        let dir = tempfile::tempdir().unwrap();
        for sub in ["memory", "memory/general"] {
            std::fs::create_dir_all(dir.path().join(sub)).unwrap();
            std::fs::write(
                dir.path().join(sub).join(crate::schema::SCHEMA_FILE_NAME),
                "fields: {}\n",
            )
            .unwrap();
        }
        let err = crate::schema::SchemaCache::build_with(
            dir.path(),
            &Default::default(),
            &Default::default(),
            &builtin_scopes(&on()),
        )
        .unwrap_err();
        let shown = err.to_string();
        assert!(shown.contains("memory/.schema.yaml"), "{shown}");
        assert!(shown.contains("memory/general/.schema.yaml"), "{shown}");
        // The same tree is fine while the group is off.
        assert!(
            crate::schema::SchemaCache::build_with(
                dir.path(),
                &Default::default(),
                &Default::default(),
                &[]
            )
            .is_ok()
        );
    }

    #[test]
    fn the_memory_area_resolves_to_the_builtin_schema_and_the_rest_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let cache = crate::schema::SchemaCache::build_with(
            dir.path(),
            &Default::default(),
            &Default::default(),
            &builtin_scopes(&on()),
        )
        .unwrap();
        let inside = cache.resolve_for(std::path::Path::new("memory/general/x.md"));
        assert!(inside.fields.contains_key("status"));
        assert!(inside.fields.get("type").is_some_and(|f| f.indexed));
        let outside = cache.resolve_for(std::path::Path::new("notes/x.md"));
        assert!(!outside.fields.contains_key("status"));
    }
}
