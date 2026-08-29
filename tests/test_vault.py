import os

import pytest

import config
import vault


class TestSlugify:
    @pytest.mark.parametrize("name", [
        "CON", "con", "Con", "NUL", "PRN", "AUX", "COM1", "COM9", "LPT1", "LPT9",
    ])
    def test_reserved_names_get_escaped(self, name):
        assert vault.slugify(name) == f"Note - {name}"

    @pytest.mark.parametrize("name", ["COM0", "COM10", "MyCON", "CONSTITUTION", "ICON"])
    def test_non_reserved_lookalikes_pass_through(self, name):
        assert vault.slugify(name) == name

    def test_reserved_name_with_padding_and_trailing_dot(self):
        assert vault.slugify(" CON ") == "Note - CON"
        assert vault.slugify("CON.") == "Note - CON"

    @pytest.mark.parametrize("text,expected", [
        ("", "Untitled"),
        ("   ", "Untitled"),
        ("___", "Untitled"),
        ("!!!???", "!!!"),  # ? and * are Windows-invalid, ! is not
        ("-", "-"),
        ("a", "a"),
    ])
    def test_empty_and_degenerate_input(self, text, expected):
        assert vault.slugify(text) == expected

    def test_custom_default(self):
        assert vault.slugify("", default="Inbox") == "Inbox"
        assert vault.slugify("   ", default="Inbox") == "Inbox"

    @pytest.mark.parametrize("char", list('<>:"/\\|?*[]#^') + ["\x00", "\x01", "\x1f"])
    def test_unsafe_chars_stripped(self, char):
        # Union of Windows-forbidden filename chars and the characters
        # Obsidian's own docs say can break wikilink parsing ([ ] # ^ | :).
        assert char not in vault.slugify(f"Idea{char}Title")

    def test_path_traversal_sequence_has_no_separators_left(self):
        slug = vault.slugify("../../Desktop")
        assert "/" not in slug and "\\" not in slug

    def test_absolute_path_has_no_separators_or_colon_left(self):
        slug = vault.slugify(r"C:\Windows\System32")
        assert "/" not in slug and "\\" not in slug and ":" not in slug

    def test_long_title_is_capped(self):
        slug = vault.slugify("A" * 5000)
        assert len(slug) <= vault.MAX_SLUG_LENGTH

    def test_capping_never_leaves_trailing_dot_or_space(self):
        slug = vault.slugify("A" * (vault.MAX_SLUG_LENGTH - 1) + ". more text")
        assert not slug.endswith(".") and not slug.endswith(" ")

    def test_rtl_and_cjk_scripts_preserved(self):
        assert vault.slugify("فكرة رائعة") == "فكرة رائعة"
        assert vault.slugify("好主意") == "好主意"

    def test_combining_marks_preserved(self):
        decomposed = "e" + "\u0301"  # e + combining acute accent
        assert vault.slugify(decomposed) == decomposed

    def test_devanagari_combining_vowels_preserved(self):
        text = "अच्छा विचार"
        assert vault.slugify(text) == text

    def test_zero_width_characters_stripped_not_glued(self):
        assert vault.slugify("Idea\u200bTitle") == "IdeaTitle"

    def test_tab_normalized_to_space(self):
        assert vault.slugify("Idea\tTitle") == "Idea Title"

    def test_emoji_is_preserved_as_a_valid_windows_filename_char(self):
        assert vault.slugify("\U0001f4a1 Idea \U0001f680") == "\U0001f4a1 Idea \U0001f680"

    def test_all_emoji_title_is_not_forced_to_fallback(self):
        text = "\U0001f4a1\U0001f680\U0001f525"
        assert vault.slugify(text) == text


class TestAppendToBody:
    def test_text_lands_above_the_first_section(self, tmp_path):
        vault.write_note(tmp_path, "project", "Foo", [], "original body", [])
        path = vault.append_to_body(tmp_path, "02_Projects/Foo.md", "a defining detail")
        text = path.read_text(encoding="utf-8")
        assert text.index("a defining detail") < text.index("## Related")
        assert "original body" in text

    def test_existing_updates_section_is_untouched(self, tmp_path):
        vault.write_note(tmp_path, "project", "Foo", [], "original body", [])
        vault.append_update(tmp_path, "02_Projects/Foo.md", "an event")
        path = vault.append_to_body(tmp_path, "02_Projects/Foo.md", "a detail")
        text = path.read_text(encoding="utf-8")
        assert "an event" in text
        assert text.index("a detail") < text.index("## Updates")

    def test_empty_text_changes_nothing(self, tmp_path):
        vault.write_note(tmp_path, "project", "Foo", [], "original body", [])
        before = (tmp_path / "02_Projects" / "Foo.md").read_text(encoding="utf-8")
        vault.append_to_body(tmp_path, "02_Projects/Foo.md", "   ")
        assert (tmp_path / "02_Projects" / "Foo.md").read_text(encoding="utf-8") == before

    def test_missing_target_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vault.append_to_body(tmp_path, "02_Projects/Nope.md", "x")


class TestRenameNote:
    def test_file_is_renamed_and_frontmatter_and_heading_follow(self, tmp_path):
        vault.write_note(tmp_path, "project", "Old Title", [], "body", [])
        path = vault.rename_note(tmp_path, "02_Projects/Old Title.md", "New Title")
        assert path.name == "New Title.md"
        assert not (tmp_path / "02_Projects" / "Old Title.md").exists()
        text = path.read_text(encoding="utf-8")
        assert 'title: "New Title"' in text
        assert "# New Title" in text

    def test_inbound_wikilinks_are_repointed(self, tmp_path):
        # Obsidian resolves links by filename, so a rename that left these
        # alone would silently break every existing reference.
        vault.write_note(tmp_path, "project", "Old Title", [], "body", [])
        vault.write_note(tmp_path, "concept", "Other", [], "body", ["Old Title"])
        vault.rename_note(tmp_path, "02_Projects/Old Title.md", "New Title")
        other = (tmp_path / "01_Concepts" / "Other.md").read_text(encoding="utf-8")
        assert "[[New Title]]" in other
        assert "[[Old Title]]" not in other

    def test_rename_onto_a_taken_filename_is_refused(self, tmp_path):
        # Merging two notes is not what a retitle asked for, and the
        # no-duplicate-filenames invariant has to hold either way.
        vault.write_note(tmp_path, "project", "Old Title", [], "body one", [])
        vault.write_note(tmp_path, "concept", "Taken", [], "body two", [])
        path = vault.rename_note(tmp_path, "02_Projects/Old Title.md", "Taken")
        assert path.name == "Old Title.md"
        assert (tmp_path / "01_Concepts" / "Taken.md").read_text(encoding="utf-8").count("body two") == 1

    def test_renaming_to_the_same_slug_is_a_noop(self, tmp_path):
        vault.write_note(tmp_path, "project", "Foo", [], "body", [])
        path = vault.rename_note(tmp_path, "02_Projects/Foo.md", "Foo")
        assert path.name == "Foo.md"

    def test_missing_target_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vault.rename_note(tmp_path, "02_Projects/Nope.md", "X")


class TestNoDuplicateFilenames:
    """A filename may never be taken twice. Reported directly: the old
    numeric-suffix disambiguation produced an "Idea Agent Project" /
    "Idea Agent Project (2)" pair -- one subject split across two files,
    with the twin unreachable by [[wikilink]] since Obsidian resolves those
    by filename."""

    def test_no_existing_note_found_in_empty_vault(self, tmp_path):
        assert vault.find_existing_note(tmp_path, "Foo") is None

    def test_finds_an_existing_note_in_its_type_folder(self, tmp_path):
        vault.write_note(tmp_path, "concept", "Foo", [], "body", [])
        found = vault.find_existing_note(tmp_path, "Foo")
        assert found is not None and found.name == "Foo.md"

    def test_finds_a_same_filename_note_in_a_different_folder(self, tmp_path):
        # Obsidian resolves [[Foo]] by filename vault-wide, so a concept Foo
        # and a project Foo are one ambiguous target, not two notes.
        vault.write_note(tmp_path, "concept", "Foo", [], "body", [])
        found = vault.find_existing_note(tmp_path, "Foo")
        assert found.parent.name == "01_Concepts"

    def test_second_write_merges_instead_of_creating_a_numbered_twin(self, tmp_path):
        first = vault.write_note(tmp_path, "project", "Idea Agent Project", [], "original body", [])
        second = vault.write_note(tmp_path, "project", "Idea Agent Project", [], "a new fact", [])
        assert second == first
        assert not (tmp_path / "02_Projects" / "Idea Agent Project (2).md").exists()
        assert len(list((tmp_path / "02_Projects").glob("*.md"))) == 1

    def test_merged_write_keeps_the_original_body_and_adds_the_new_one(self, tmp_path):
        vault.write_note(tmp_path, "project", "Foo", [], "original body", [])
        path = vault.write_note(tmp_path, "project", "Foo", [], "a new fact", [])
        text = path.read_text(encoding="utf-8")
        assert "original body" in text  # never overwritten
        assert "a new fact" in text
        assert "## Updates" in text

    def test_merge_happens_even_across_different_types(self, tmp_path):
        vault.write_note(tmp_path, "concept", "Foo", [], "original body", [])
        path = vault.write_note(tmp_path, "project", "Foo", [], "a new fact", [])
        assert path.parent.name == "01_Concepts"  # merged into the existing one
        assert not (tmp_path / "02_Projects" / "Foo.md").exists()

    @pytest.mark.skipif(os.name != "nt", reason="depends on a case-insensitive filesystem")
    def test_case_differing_title_merges_on_windows(self, tmp_path):
        vault.write_note(tmp_path, "concept", "My Idea", [], "original body", [])
        vault.write_note(tmp_path, "concept", "my idea", [], "a new fact", [])
        assert len(list((tmp_path / "01_Concepts").glob("*.md"))) == 1


class TestWriteNote:
    def test_basic_write_has_expected_frontmatter_and_body(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Water Tracker", ["health", "water"],
                                 "Track water intake.", [])
        text = path.read_text(encoding="utf-8")
        assert 'title: "Water Tracker"' in text
        assert "type: concept" in text
        assert "- health" in text and "- water" in text
        assert "# Water Tracker" in text
        assert "Track water intake." in text
        assert "none yet" in text

    @pytest.mark.parametrize("note_type,folder", list(config.FOLDER_BY_TYPE.items()))
    def test_each_type_routes_to_its_mandated_folder(self, tmp_path, note_type, folder):
        path = vault.write_note(tmp_path, note_type, "X", [], "body", [])
        assert path.parent.name == folder

    def test_unknown_type_falls_back_to_log_folder(self, tmp_path):
        path = vault.write_note(tmp_path, "not-a-real-type", "X", [], "body", [])
        assert path.parent.name == config.FOLDER_BY_TYPE["log"]

    def test_links_become_wikilinks(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Idea B", [], "body", ["Idea A"])
        assert "- [[Idea A]]" in path.read_text(encoding="utf-8")

    def test_title_with_quotes_is_escaped_safely(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", 'A "quoted" title', [], "body", [])
        text = path.read_text(encoding="utf-8")
        assert 'title: "A \\"quoted\\" title"' in text

    def test_very_long_title_does_not_crash(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "A" * 5000, [], "body", [])
        assert path.exists()
        assert len(path.stem) <= vault.MAX_SLUG_LENGTH

    def test_long_title_collision_does_not_crash(self, tmp_path):
        long_title = "A" * 5000
        vault.write_note(tmp_path, "concept", long_title, [], "body", [])
        second = vault.write_note(tmp_path, "concept", long_title, [], "body 2", [])
        assert second.exists()

    def test_newline_in_title_does_not_break_heading(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Line1\nLine2", [], "body", [])
        text = path.read_text(encoding="utf-8")
        assert "# Line1 Line2" in text

    def test_duplicate_tags_are_deduplicated(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Idea", ["idea", "idea"], "body", [])
        text = path.read_text(encoding="utf-8")
        assert text.count("- idea") == 1

    def test_unicode_tags_round_trip(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Idea", ["café", "日本語"], "body", [])
        text = path.read_text(encoding="utf-8")
        assert "café" in text and "日本語" in text

    def test_no_links_shows_placeholder(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Idea", [], "body", [])
        assert "- none yet" in path.read_text(encoding="utf-8")

    def test_tag_list_is_indented_matching_obsidians_own_convention(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Idea", ["a", "b"], "body", [])
        text = path.read_text(encoding="utf-8")
        assert "tags:\n  - a\n  - b" in text

    def test_link_target_matches_actual_filename_when_title_has_unsafe_chars(self, tmp_path):
        # The file is saved under slugify(title); a wikilink using the raw
        # title instead would silently fail to resolve in Obsidian.
        messy_title = "Idea: with #special chars"
        linked = vault.write_note(tmp_path, "concept", messy_title, [], "body", [])
        referrer = vault.write_note(tmp_path, "concept", "Referrer", [], "body", [messy_title])
        text = referrer.read_text(encoding="utf-8")
        assert f"[[{linked.stem}|{messy_title}]]" in text
        assert linked.stem == vault.slugify(messy_title)

    def test_link_target_is_plain_when_title_needs_no_sanitizing(self, tmp_path):
        path = vault.write_note(tmp_path, "concept", "Idea B", [], "body", ["Clean Title"])
        text = path.read_text(encoding="utf-8")
        assert "- [[Clean Title]]" in text
        assert "|" not in text.split("## Related")[1]

    def test_frontmatter_is_valid_yaml(self, tmp_path):
        import yaml
        path = vault.write_note(tmp_path, "concept", "Idea: with colon", ["a-tag"], "body", ["Other"])
        text = path.read_text(encoding="utf-8")
        block = text.split("---\n")[1]
        parsed = yaml.safe_load(block)
        assert parsed["type"] == "concept"
        assert parsed["tags"] == ["a-tag"]


class TestAppendDailyLogLinks:
    def test_creates_log_note_on_first_use(self, tmp_path):
        path = vault.append_daily_log_links(tmp_path, ["New Note"], [])
        assert path.exists()
        assert path.parent.name == config.FOLDER_BY_TYPE["log"]

    def test_new_note_titles_get_linked(self, tmp_path):
        path = vault.append_daily_log_links(tmp_path, ["Note A", "Note B"], [])
        text = path.read_text(encoding="utf-8")
        assert "- [[Note A]]" in text and "- [[Note B]]" in text

    def test_duplicates_get_logged_with_link_to_existing(self, tmp_path):
        path = vault.append_daily_log_links(tmp_path, [], [("raw capture text", "Existing Note")])
        text = path.read_text(encoding="utf-8")
        assert "[[Existing Note]]" in text
        assert "raw capture text" in text

    def test_second_call_same_day_appends_not_overwrites(self, tmp_path):
        vault.append_daily_log_links(tmp_path, ["First"], [])
        path = vault.append_daily_log_links(tmp_path, ["Second"], [])
        text = path.read_text(encoding="utf-8")
        assert "[[First]]" in text and "[[Second]]" in text

    def test_no_new_content_does_not_error(self, tmp_path):
        path = vault.append_daily_log_links(tmp_path, [], [])
        assert path.exists()

    def test_updated_notes_get_logged_with_link(self, tmp_path):
        path = vault.append_daily_log_links(tmp_path, [], [], [("raw capture text", "Existing Note")])
        text = path.read_text(encoding="utf-8")
        assert "[[Existing Note]]" in text
        assert "raw capture text" in text


class TestDeleteNote:
    def test_sends_to_recycle_bin_not_permanent_delete(self, tmp_path, mocker):
        note = tmp_path / "Existing.md"
        note.write_text("# Existing\n", encoding="utf-8")
        spy = mocker.patch("vault.send2trash.send2trash")
        vault.delete_note(tmp_path, "Existing.md")
        spy.assert_called_once_with(str(note))

    def test_missing_target_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vault.delete_note(tmp_path, "Does Not Exist.md")


class TestAddLink:
    def test_adds_link_under_related_heading(self, tmp_path):
        note = tmp_path / "Source.md"
        note.write_text("# Source\n\nBody.\n\n## Related\n- none yet\n", encoding="utf-8")
        vault.add_link(tmp_path, "Source.md", "Target Note")
        text = note.read_text(encoding="utf-8")
        assert "[[Target Note]]" in text

    def test_creates_related_heading_if_missing(self, tmp_path):
        note = tmp_path / "Source.md"
        note.write_text("# Source\n\nBody.\n", encoding="utf-8")
        vault.add_link(tmp_path, "Source.md", "Target Note")
        text = note.read_text(encoding="utf-8")
        assert "## Related" in text and "[[Target Note]]" in text

    def test_idempotent_does_not_duplicate_existing_link(self, tmp_path):
        note = tmp_path / "Source.md"
        note.write_text("# Source\n\n## Related\n- [[Target Note]]\n", encoding="utf-8")
        vault.add_link(tmp_path, "Source.md", "Target Note")
        text = note.read_text(encoding="utf-8")
        assert text.count("[[Target Note]]") == 1

    def test_missing_source_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vault.add_link(tmp_path, "Does Not Exist.md", "Target Note")

    def test_nothing_existing_is_touched(self, tmp_path):
        note = tmp_path / "Source.md"
        original = "# Source\n\nOriginal body content.\n"
        note.write_text(original, encoding="utf-8")
        vault.add_link(tmp_path, "Source.md", "Target Note")
        text = note.read_text(encoding="utf-8")
        assert text.startswith(original.rstrip("\n"))


class TestAppendUpdate:
    """The one operation in this codebase that touches an existing note at
    all -- deliberately narrow: read full content, insert exactly one new
    bullet, write full content back. Never rewrites, reorders, or removes
    anything already there."""

    def test_adds_updates_heading_on_first_use(self, tmp_path):
        note = tmp_path / "Existing.md"
        note.write_text("---\ntitle: \"Existing\"\n---\n\n# Existing\n\nOriginal body.\n", encoding="utf-8")
        vault.append_update(tmp_path, "Existing.md", "a new fact")
        text = note.read_text(encoding="utf-8")
        assert "## Updates" in text
        assert "a new fact" in text
        assert "Original body." in text  # nothing existing was touched

    def test_second_update_appends_under_same_heading_not_a_new_one(self, tmp_path):
        note = tmp_path / "Existing.md"
        note.write_text("# Existing\n\nBody.\n", encoding="utf-8")
        vault.append_update(tmp_path, "Existing.md", "first fact")
        vault.append_update(tmp_path, "Existing.md", "second fact")
        text = note.read_text(encoding="utf-8")
        assert text.count("## Updates") == 1
        assert "first fact" in text and "second fact" in text

    def test_missing_target_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vault.append_update(tmp_path, "Does Not Exist.md", "a fact")

    def test_bullet_includes_todays_date(self, tmp_path):
        import datetime
        note = tmp_path / "Existing.md"
        note.write_text("# Existing\n\nBody.\n", encoding="utf-8")
        vault.append_update(tmp_path, "Existing.md", "a fact")
        text = note.read_text(encoding="utf-8")
        assert datetime.date.today().isoformat() in text

    def test_frontmatter_and_heading_survive_untouched(self, tmp_path):
        note = tmp_path / "Existing.md"
        original = '---\ntitle: "Existing"\ntype: project\ntags:\n  - a\n---\n\n# Existing\n\nBody.\n'
        note.write_text(original, encoding="utf-8")
        vault.append_update(tmp_path, "Existing.md", "a fact")
        text = note.read_text(encoding="utf-8")
        assert text.startswith(original.rstrip("\n"))
