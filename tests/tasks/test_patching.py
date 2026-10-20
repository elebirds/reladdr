"""验证干预样本的真值可区分、参照保持有效及场景族划分。"""

from reladdr.tasks.patching import (
    PAPER_PALETTE,
    make_e02_pairs,
    make_e03_pairs,
    make_reference_move_pairs,
)

COUNTS = {"discovery": 2, "validation": 1, "test": 1}


def test_e02_labels_are_distinct_and_donor_preserves_reference():
    pairs = make_e02_pairs(
        19, COUNTS, ["squares_horizontal", "shapes_horizontal", "shapes_vertical"]
    )
    assert len(pairs) == 4 * 4 * 3
    for pair in pairs:
        assert len({pair.clean_label, pair.address_label, pair.attribute_label}) == 3
        assert pair.recipient.reference == pair.donor.reference
        assert pair.recipient.objects[1] == pair.donor.objects[1]
        assert pair.recipient.target_index != pair.donor.target_index
        assert pair.address_label == pair.recipient.objects[pair.donor.target_index].color
        assert pair.recipient.image().size == (pair.recipient.size, pair.recipient.size)


def test_paper_geometry_and_palette_match_source():
    pair = make_e02_pairs(19, COUNTS, ["squares_horizontal"])[0]
    assert pair.recipient.size == 336
    assert pair.recipient.objects[0].bbox == (28, 140, 84, 196)
    assert pair.recipient.objects[-1].bbox == (252, 140, 308, 196)
    assert set(PAPER_PALETTE) == {"red", "green", "blue", "black", "brown", "gray"}
    assert (
        pair.recipient.image().getpixel((28, 140)) == PAPER_PALETTE[pair.recipient.objects[0].color]
    )
    assert pair.recipient.image().getpixel((84, 140)) == (255, 255, 255)


def test_family_split_stable_when_expanded():
    first = make_e02_pairs(19, COUNTS, ["squares_horizontal"])
    expanded = make_e02_pairs(
        19, {"discovery": 3, "validation": 1, "test": 1}, ["squares_horizontal"]
    )
    by_id = {p.pair_id: p for p in expanded}
    assert all(by_id[p.pair_id] == p for p in first)
    assert all(
        len({p.split for p in first if p.family_id == family}) == 1
        for family in {p.family_id for p in first}
    )


def test_e03_changes_only_reference_in_same_relation_and_image():
    pairs = make_e03_pairs(19, COUNTS)
    assert {p.recipient.layout for p in pairs} == {"horizontal", "grid2"}
    for pair in pairs:
        assert pair.recipient.objects == pair.donor.objects
        assert pair.recipient.relation == pair.donor.relation
        assert pair.recipient.reference != pair.donor.reference
        assert pair.clean_label != pair.address_label
        assert len({o.color for o in pair.recipient.objects}) == len(pair.recipient.objects)


def test_no_paper_clean_or_donor_image_crosses_family_or_split():
    pairs = make_e02_pairs(19, {"discovery": 8, "validation": 4, "test": 4}, ["squares_horizontal"])
    assigned = {}
    for pair in pairs:
        for item in (pair.recipient, pair.donor):
            signature = tuple(obj.color for obj in item.objects)
            previous = assigned.setdefault(signature, (item.family_id, item.split))
            assert previous == (item.family_id, item.split)


def test_same_instruction_moves_reference_not_instruction_and_preserves_distinct_labels():
    counts = {"discovery": 8, "validation": 4, "test": 4}
    pairs = make_reference_move_pairs(19, counts)
    assert len(pairs) == 32
    owners = {}
    for pair in make_e02_pairs(19, counts, ["squares_horizontal"]) + pairs:
        for item in (pair.recipient, pair.donor):
            signature = tuple(obj.color for obj in item.objects)
            assert owners.setdefault(signature, item.family_id) == item.family_id
    for pair in pairs:
        assert pair.recipient.prompt == pair.donor.prompt
        assert pair.recipient.relation == pair.donor.relation
        assert pair.recipient.reference == pair.donor.reference
        assert len({pair.clean_label, pair.address_label, pair.attribute_label}) == 3
        assert pair.recipient.objects[1].color in {
            pair.donor.objects[0].color,
            pair.donor.objects[2].color,
        }


def test_adjacent_reference_move_makes_relation_unambiguous():
    pairs = make_reference_move_pairs(19, COUNTS, adjacent=True)
    assert all("immediately" in p.recipient.prompt for p in pairs)
    assert all(p.recipient.prompt == p.donor.prompt for p in pairs)


def test_e02_adjacent_control_only_changes_prompt_not_images_or_labels():
    first = make_e02_pairs(19, COUNTS, ["squares_horizontal"])
    matched = make_e02_pairs(19, COUNTS, ["squares_horizontal"], adjacent=True)
    for original, changed in zip(first, matched, strict=True):
        assert original.recipient.objects == changed.recipient.objects
        assert original.donor.objects == changed.donor.objects
        assert (original.clean_label, original.address_label, original.attribute_label) == (
            changed.clean_label,
            changed.address_label,
            changed.attribute_label,
        )
        assert "immediately" in changed.recipient.prompt
        assert "immediately" in changed.donor.prompt


def test_grid_control_removes_reference_label_overlap_and_keeps_disjoint_images():
    from reladdr.tasks.patching import make_grid_move_pairs

    pairs = make_grid_move_pairs(19, {"discovery": 8, "validation": 4, "test": 4})
    assert len(pairs) == 64
    owners = {}
    for pair in pairs:
        assert pair.recipient.layout == "grid2"
        assert pair.recipient.prompt == pair.donor.prompt
        assert pair.recipient.reference == pair.donor.reference
        reference_color = pair.recipient.reference.split()[0]
        assert (
            len({pair.clean_label, pair.address_label, pair.attribute_label, reference_color}) == 4
        )
        assert pair.address_label == pair.recipient.objects[pair.donor.target_index].color
        assert pair.attribute_label not in {obj.color for obj in pair.recipient.objects}
        for item in (pair.recipient, pair.donor):
            signature = tuple(obj.color for obj in item.objects)
            assert owners.setdefault(signature, item.family_id) == item.family_id
            assert len(set(signature)) == 4
