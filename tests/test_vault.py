import pytest

import vault


class TestSlugify:
    @pytest.mark.parametrize("name", [
        "CON", "con", "Con", "NUL", "PRN", "AUX", "COM1", "COM9", "LPT1", "LPT9",
    ])
    def test_reserved_names_get_escaped(self, name):
        assert vault.slugify(name) == f"Idea - {name}"

    @pytest.mark.parametrize("name", ["COM0", "COM10", "MyCON", "CONSTITUTION", "ICON"])
    def test_non_reserved_lookalikes_pass_through(self, name):
        assert vault.slugify(name) == name

    def test_reserved_name_with_padding_and_trailing_dot(self):
        assert vault.slugify(" CON ") == "Idea - CON"
        assert vault.slugify("CON.") == "Idea - CON"

    @pytest.mark.parametrize("text,expected", [
        ("", "Untitled Idea"),
        ("   ", "Untitled Idea"),
        ("___", "Untitled Idea"),
        ("!!!???", "!!!"),  # ? and * are Windows-invalid, ! is not
        ("-", "-"),
        ("a", "a"),
    ])
    def test_empty_and_degenerate_input(self, text, expected):
        assert vault.slugify(text) == expected

    def test_custom_default(self):
        assert vault.slugify("", default="Inbox") == "Inbox"
        assert vault.slugify("   ", default="Inbox") == "Inbox"

    @pytest.mark.parametrize("char", list('<>:"/\\|?*') + ["\x00", "\x01", "\x1f"])
    def test_windows_invalid_chars_stripped(self, char):
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
        # Emoji are valid in Windows/NTFS filenames, so the blacklist-based
        # sanitizer intentionally leaves them alone rather than guessing at
        # what counts as "real content" -- that's what let Devanagari/RTL/CJK
        # text through correctly too.
        assert vault.slugify("\U0001f4a1 Idea \U0001f680") == "\U0001f4a1 Idea \U0001f680"

    def test_all_emoji_title_is_not_forced_to_fallback(self):
        text = "\U0001f4a1\U0001f680\U0001f525"
        assert vault.slugify(text) == text


class TestUniquePath:
    def test_no_collision_returns_bare_slug(self, tmp_path):
        assert vault.unique_path(tmp_path, "Foo").name == "Foo.md"

    def test_collision_sequence(self, tmp_path):
        (tmp_path / "Foo.md").touch()
        (tmp_path / "Foo (2).md").touch()
        (tmp_path / "Foo (3).md").touch()
        assert vault.unique_path(tmp_path, "Foo").name == "Foo (4).md"

    def test_gap_in_sequence_fills_the_gap(self, tmp_path):
        (tmp_path / "Foo (2).md").touch()
        assert vault.unique_path(tmp_path, "Foo").name == "Foo.md"

    def test_case_insensitive_collision_on_windows(self, tmp_path):
        (tmp_path / "My Idea.md").touch()
        result = vault.unique_path(tmp_path, "my idea")
        assert result.name != "my idea.md"


class TestWriteNote:
    def test_basic_write_has_expected_frontmatter_and_body(self, tmp_path):
        path = vault.write_note(tmp_path, "Health", "Water Tracker", ["health", "water"],
                                 "Track water intake.", [])
        text = path.read_text(encoding="utf-8")
        assert "type: idea" in text
        assert "- health" in text and "- water" in text
        assert "# Water Tracker" in text
        assert "Track water intake." in text
        assert "none yet" in text

    def test_related_titles_become_wikilinks(self, tmp_path):
        path = vault.write_note(tmp_path, "Health", "Idea B", [], "body", ["Idea A"])
        assert "- [[Idea A]]" in path.read_text(encoding="utf-8")

    def test_folder_traversal_is_contained(self, tmp_path):
        vault_root = tmp_path / "Vault"
        vault_root.mkdir()
        path = vault.write_note(vault_root, "../../Desktop", "test idea", [], "body", [])
        assert vault_root.resolve() in path.resolve().parents

    def test_absolute_path_folder_is_contained(self, tmp_path):
        vault_root = tmp_path / "Vault"
        vault_root.mkdir()
        path = vault.write_note(vault_root, r"C:\Windows\System32", "test idea", [], "body", [])
        assert vault_root.resolve() in path.resolve().parents

    def test_rooted_folder_is_contained(self, tmp_path):
        vault_root = tmp_path / "Vault"
        vault_root.mkdir()
        path = vault.write_note(vault_root, "/Ideas", "test idea", [], "body", [])
        assert vault_root.resolve() in path.resolve().parents

    def test_unc_style_folder_is_contained(self, tmp_path):
        vault_root = tmp_path / "Vault"
        vault_root.mkdir()
        path = vault.write_note(vault_root, r"\\server\share\folder", "test idea", [], "body", [])
        assert vault_root.resolve() in path.resolve().parents

    def test_reserved_folder_name_is_escaped(self, tmp_path):
        path = vault.write_note(tmp_path, "CON", "test idea", [], "body", [])
        assert "Idea - CON" in path.parts

    def test_empty_folder_defaults_to_inbox(self, tmp_path):
        path = vault.write_note(tmp_path, "   ", "test idea", [], "body", [])
        assert path.parent.name == "Inbox"

    def test_very_long_title_does_not_crash(self, tmp_path):
        path = vault.write_note(tmp_path, "Health", "A" * 5000, [], "body", [])
        assert path.exists()
        assert len(path.stem) <= vault.MAX_SLUG_LENGTH

    def test_long_title_collision_does_not_crash(self, tmp_path):
        long_title = "A" * 5000
        vault.write_note(tmp_path, "Health", long_title, [], "body", [])
        second = vault.write_note(tmp_path, "Health", long_title, [], "body 2", [])
        assert second.exists()

    def test_newline_in_title_does_not_break_heading(self, tmp_path):
        path = vault.write_note(tmp_path, "Health", "Line1\nLine2", [], "body", [])
        text = path.read_text(encoding="utf-8")
        assert "# Line1 Line2" in text

    def test_duplicate_tags_are_deduplicated(self, tmp_path):
        path = vault.write_note(tmp_path, "Health", "Idea", ["idea", "idea"], "body", [])
        text = path.read_text(encoding="utf-8")
        assert text.count("- idea") == 1

    def test_unicode_tags_round_trip(self, tmp_path):
        path = vault.write_note(tmp_path, "Health", "Idea", ["café", "日本語"], "body", [])
        text = path.read_text(encoding="utf-8")
        assert "café" in text and "日本語" in text

    def test_no_related_titles_shows_placeholder(self, tmp_path):
        path = vault.write_note(tmp_path, "Health", "Idea", [], "body", [])
        assert "- none yet" in path.read_text(encoding="utf-8")
