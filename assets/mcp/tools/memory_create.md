Save one memory: a short, self-contained statement worth recalling later (a
preference, fact, decision, finding or procedure). The server writes the file,
its id and its provenance; you supply only the content.

If a live memory already says nearly the same thing, nothing is written and the
matches come back as `duplicates` (`created: false`): update one with
`memory_update`, or call again with `on_duplicate: supersede` to replace the
closest, or `keep_both` to save it anyway.
