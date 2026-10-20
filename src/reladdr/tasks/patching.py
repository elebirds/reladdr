"""构造 E02 的三标签可区分 donor 对及 E03 的同关系变参照场景，固定族划分。"""

import random
from dataclasses import asdict, dataclass, replace
from itertools import permutations
from types import SimpleNamespace

from PIL import Image, ImageDraw

from reladdr.tasks.relational import make_grouped_scenes
from reladdr.tasks.synthetic import PALETTE, SceneObject, render_objects

# 与 R3 公开 shared/palette.py、experiment_config.py 对齐；不修改旧 E01 调色板。
PAPER_PALETTE = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "brown": (150, 75, 0),
    "gray": (128, 128, 128),
}
PREPOSITIONS = {
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}


@dataclass(frozen=True)
class PatchingInput:
    input_id: str
    family_id: str
    split: str
    setting: str
    layout: str
    prompt: str
    answer: str
    reference: str
    relation: str
    target_index: int
    objects: tuple[SceneObject, ...]
    size: int
    palette: str

    def record(self):
        return asdict(self)

    def image(self):
        if self.palette == "project":
            return render_objects(self.objects, self.size)
        canvas = Image.new("RGB", (self.size, self.size), "white")
        draw = ImageDraw.Draw(canvas)
        for obj in self.objects:
            x0, y0, x1, y1 = obj.bbox
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=PAPER_PALETTE[obj.color])
        return canvas


@dataclass(frozen=True)
class PatchingPair:
    pair_id: str
    family_id: str
    split: str
    contrast: str
    recipient: PatchingInput
    donor: PatchingInput
    clean_label: str
    address_label: str
    attribute_label: str

    def record(self):
        return asdict(self)


def _input(
    scene, objects, *, input_id, setting, relation, target_index, palette, size, reference_index=1
):
    reference_obj = objects[reference_index]
    reference = f"{reference_obj.color} {reference_obj.shape}"
    noun = "square" if palette == "paper" else "shape"
    prompt = f"The color of the {noun} {PREPOSITIONS[relation]} the {reference} is "
    return PatchingInput(
        input_id,
        scene.family_id,
        scene.split,
        setting,
        scene.layout,
        prompt,
        objects[target_index].color,
        reference,
        relation,
        target_index,
        tuple(objects),
        size,
        palette,
    )


def make_e02_pairs(seed, family_counts, settings, *, adjacent=False):
    allowed = {"squares_horizontal", "shapes_horizontal", "shapes_vertical"}
    if not settings or len(settings) != len(set(settings)) or set(settings) - allowed:
        raise ValueError("E02 setting 必须已知且不重复")
    pairs = []
    if "squares_horizontal" in settings:
        pairs.extend(_paper_pairs(seed, family_counts))
    scenes = make_grouped_scenes(seed, family_counts, ["horizontal", "vertical"])
    for scene in scenes:
        setting = "shapes_" + scene.layout
        if setting not in settings:
            continue
        objects = scene.objects
        remaining = [color for color in PALETTE if color not in {obj.color for obj in objects}]
        directions = ("left", "right") if scene.layout == "horizontal" else ("above", "below")
        for variant in range(2):
            donor_objects = list(objects)
            donor_objects[0] = replace(objects[0], color=remaining[variant % 3])
            donor_objects[2] = replace(objects[2], color=remaining[(variant + 1) % 3])
            for relation in directions:
                target = 0 if relation in {"left", "above"} else 2
                pair_id = f"{scene.scene_id}-{setting}-v{variant}-{relation}"
                recipient = _input(
                    scene,
                    objects,
                    input_id=pair_id + "-clean",
                    setting=setting,
                    relation=relation,
                    target_index=target,
                    palette="project",
                    size=scene.size,
                )
                donor = _input(
                    scene,
                    donor_objects,
                    input_id=pair_id + "-donor",
                    setting=setting,
                    relation=OPPOSITE[relation],
                    target_index=2 - target,
                    palette="project",
                    size=scene.size,
                )
                labels = (recipient.answer, objects[donor.target_index].color, donor.answer)
                if len(set(labels)) != 3:
                    raise ValueError("E02 的原答案、地址迁移、属性迁移标签必须两两不同")
                pairs.append(
                    PatchingPair(
                        pair_id,
                        scene.family_id,
                        scene.split,
                        "direction_and_image",
                        recipient,
                        donor,
                        *labels,
                    )
                )
    if adjacent:

        def with_adjacent(item):
            phrase = PREPOSITIONS[item.relation]
            return replace(
                item, prompt=item.prompt.replace(f" {phrase} ", f" immediately {phrase} ", 1)
            )

        pairs = [
            replace(pair, recipient=with_adjacent(pair.recipient), donor=with_adjacent(pair.donor))
            for pair in pairs
        ]
    return pairs


def _paper_pairs(seed, family_counts):
    # 固定几何会抹去 Shapes 的身份差异。独立分配 clean/donor 图像束，不能沿用可能
    # 重合的 E01 颜色顺序并伪装为不同族，更不能用微小平移掩盖内容重复。
    anchors = list(permutations(PAPER_PALETTE, 3))
    random.Random(seed).shuffle(anchors)
    used = set()
    bundles = []
    for clean in anchors:
        if clean in used:
            continue
        others = [c for c in PAPER_PALETTE if c not in clean]
        donors = [
            (a, clean[1], b) for a, b in permutations(others, 2) if (a, clean[1], b) not in used
        ]
        if len(donors) < 2:
            continue
        bundle = (clean, *donors[:2])
        bundles.append(bundle)
        used.update(bundle)
    n = len(bundles)
    assignments = (
        ["discovery"] * (n - 2 * (n // 5)) + ["validation"] * (n // 5) + ["test"] * (n // 5)
    )
    random.Random(seed + 913).shuffle(assignments)
    if set(family_counts) != {"discovery", "validation", "test"} or any(
        type(v) is not int or v < 1 for v in family_counts.values()
    ):
        raise ValueError("必须指定三个正整数场景族数量")
    if any(family_counts[split] > assignments.count(split) for split in family_counts):
        raise ValueError("请求超出固定 Squares 图像束池的 split 容量；扩量需修订协议")
    selected = {split: 0 for split in family_counts}
    result = []
    for family_index, (bundle, split) in enumerate(zip(bundles, assignments, strict=True)):
        if selected[split] >= family_counts[split]:
            continue
        selected[split] += 1
        scene = SimpleNamespace(
            family_id=f"p{seed}-family-{family_index:04d}", split=split, layout="horizontal"
        )
        inventories = [
            tuple(
                SceneObject(
                    f"object-{i:02d}", "square", color, (28 + i * 112, 140, 84 + i * 112, 196)
                )
                for i, color in enumerate(colors)
            )
            for colors in bundle
        ]
        for variant, donor_objects in enumerate(inventories[1:]):
            for relation, target in (("left", 0), ("right", 2)):
                pair_id = f"{scene.family_id}-squares_horizontal-v{variant}-{relation}"
                recipient = _input(
                    scene,
                    inventories[0],
                    input_id=pair_id + "-clean",
                    setting="squares_horizontal",
                    relation=relation,
                    target_index=target,
                    palette="paper",
                    size=336,
                )
                donor = _input(
                    scene,
                    donor_objects,
                    input_id=pair_id + "-donor",
                    setting="squares_horizontal",
                    relation=OPPOSITE[relation],
                    target_index=2 - target,
                    palette="paper",
                    size=336,
                )
                labels = (
                    recipient.answer,
                    recipient.objects[donor.target_index].color,
                    donor.answer,
                )
                result.append(
                    PatchingPair(
                        pair_id,
                        scene.family_id,
                        split,
                        "direction_and_image",
                        recipient,
                        donor,
                        *labels,
                    )
                )
    return result


def make_e03_pairs(seed, family_counts, layouts=("horizontal", "grid2")):
    """同图、同关系、改变参照；颜色唯一，先排除答案到对象的多义映射。"""
    pairs = []
    scenes = make_grouped_scenes(seed, family_counts, list(layouts))
    for scene in scenes:
        rng = random.Random(seed + int(scene.family_id.rsplit("-", 1)[-1]) * 39)
        colors = rng.sample(list(PAPER_PALETTE), scene.rows * scene.columns)
        if scene.layout == "horizontal":
            boxes = [(28 + i * 112, 140, 84 + i * 112, 196) for i in range(3)]
        else:
            boxes = [
                (56 + c * 168, 56 + r * 168, 112 + c * 168, 112 + r * 168)
                for r in range(2)
                for c in range(2)
            ]
        objects = tuple(
            SceneObject(f"object-{i:02d}", "square", color, box)
            for i, (color, box) in enumerate(zip(colors, boxes, strict=True))
        )
        directions = {"left": (0, -1), "right": (0, 1), "above": (-1, 0), "below": (1, 0)}
        for relation, (dr, dc) in directions.items():
            valid = []
            for reference_index in range(len(objects)):
                r, c = divmod(reference_index, scene.columns)
                if 0 <= r + dr < scene.rows and 0 <= c + dc < scene.columns:
                    valid.append((reference_index, (r + dr) * scene.columns + c + dc))
            if len(valid) < 2:
                continue
            for index, (ref, target) in enumerate(valid):
                other_ref, other_target = valid[(index + 1) % len(valid)]
                pair_id = f"{scene.scene_id}-same-{relation}-ref{ref}-to{other_ref}"
                recipient = _input(
                    scene,
                    objects,
                    input_id=pair_id + "-clean",
                    setting="squares_dynamic_reference",
                    relation=relation,
                    target_index=target,
                    palette="paper",
                    size=336,
                    reference_index=ref,
                )
                donor = _input(
                    scene,
                    objects,
                    input_id=pair_id + "-donor",
                    setting="squares_dynamic_reference",
                    relation=relation,
                    target_index=other_target,
                    palette="paper",
                    size=336,
                    reference_index=other_ref,
                )
                pairs.append(
                    PatchingPair(
                        pair_id,
                        scene.family_id,
                        scene.split,
                        "reference_only",
                        recipient,
                        donor,
                        recipient.answer,
                        donor.answer,
                        donor.answer,
                    )
                )
    return pairs


def make_reference_move_pairs(seed, family_counts, *, adjacent=False):
    """保持逐字相同的问题，跨图改变参照位置，检验 E02 端口是否只迁移了方向指令。"""
    source = make_e02_pairs(seed, family_counts, ["squares_horizontal"])
    owners = {}
    recipients = {}
    for pair in source:
        recipients.setdefault(pair.family_id, pair.recipient)
        for item in (pair.recipient, pair.donor):
            owners[tuple(obj.color for obj in item.objects)] = item.family_id
    result = []
    for family_index, original in enumerate(recipients.values()):
        reference_color = original.objects[1].color
        for relation, clean_target, donor_reference in (("left", 0, 2), ("right", 2, 0)):
            preposition = ("immediately " if adjacent else "") + PREPOSITIONS[relation]
            pair_id = f"{original.family_id}-same-instruction-refmove-{relation}"
            recipient = replace(
                original,
                input_id=pair_id + "-clean",
                relation=relation,
                target_index=clean_target,
                answer=original.objects[clean_target].color,
                prompt=(f"The color of the square {preposition} the {reference_color} square is "),
            )
            choices = list(
                permutations([color for color in PAPER_PALETTE if color != reference_color], 2)
            )
            random.Random(seed + family_index * 71 + clean_target).shuffle(choices)
            chosen = None
            for target_color, other_color in choices:
                if target_color in {recipient.answer, reference_color}:
                    continue
                inventory = [other_color, target_color, other_color]
                inventory[donor_reference] = reference_color
                signature = tuple(inventory)
                if signature in owners and owners[signature] != original.family_id:
                    continue
                chosen = inventory
                owners[signature] = original.family_id
                break
            if chosen is None:
                raise ValueError("没有无跨族/跨划分重复的参照移位 donor；不能静默复用图像")
            donor_objects = tuple(
                replace(obj, color=chosen[i]) for i, obj in enumerate(original.objects)
            )
            donor = replace(
                recipient,
                input_id=pair_id + "-donor",
                objects=donor_objects,
                target_index=1,
                answer=donor_objects[1].color,
            )
            labels = (recipient.answer, recipient.objects[1].color, donor.answer)
            if len(set(labels)) != 3 or recipient.prompt != donor.prompt:
                raise ValueError("同指令移位对必须逐字一致且三个结果可区分")
            result.append(
                PatchingPair(
                    pair_id,
                    original.family_id,
                    original.split,
                    "reference_position_same_instruction",
                    recipient,
                    donor,
                    *labels,
                )
            )
    return result


def make_grid_move_pairs(seed, family_counts):
    """用二维格排除一维反例中“地址标签恰为参照颜色”的结构性混淆。"""
    source = make_e02_pairs(seed, family_counts, ["squares_horizontal"])
    families = {pair.family_id: pair.split for pair in source}
    inventories = list(permutations(PAPER_PALETTE, 4))
    random.Random(seed + 4103).shuffle(inventories)
    boxes = [
        (56 + c * 168, 56 + r * 168, 112 + c * 168, 112 + r * 168)
        for r in range(2)
        for c in range(2)
    ]
    owners = {}
    original = {}
    for family_id, _split in families.items():
        index = int(family_id.rsplit("-", 1)[-1])
        colors = inventories[index]
        original[family_id] = tuple(
            SceneObject(f"object-{i:02d}", "square", color, boxes[i])
            for i, color in enumerate(colors)
        )
        owners[colors] = family_id
    result = []
    for family_id, split in families.items():
        index = int(family_id.rsplit("-", 1)[-1])
        objects = original[family_id]
        # 每方向固定一对，同一个族的四个方向不跨划分；奇偶族交替起始参照格。
        layout = {
            "left": (1 + 2 * (index % 2), 3 - 2 * (index % 2), -1),
            "right": (2 * (index % 2), 2 - 2 * (index % 2), 1),
            "above": (2 + index % 2, 3 - index % 2, -2),
            "below": (index % 2, 1 - index % 2, 2),
        }
        scene = SimpleNamespace(family_id=family_id, split=split, layout="grid2")
        for offset, (relation, (reference_index, donor_reference, delta)) in enumerate(
            layout.items()
        ):
            target = reference_index + delta
            donor_target = donor_reference + delta
            reference_color = objects[reference_index].color
            b = objects[donor_target].color
            if b == reference_color:
                raise ValueError("二维控制不能让地址标签等于参照颜色")
            choices = []
            for c in PAPER_PALETTE:
                if c in {obj.color for obj in objects}:
                    continue
                for others in permutations(
                    [color for color in PAPER_PALETTE if color not in {c, reference_color}], 2
                ):
                    colors = [None] * 4
                    colors[donor_reference] = reference_color
                    colors[donor_target] = c
                    for slot, color in zip(
                        [i for i in range(4) if colors[i] is None], others, strict=True
                    ):
                        colors[slot] = color
                    if tuple(colors) not in owners or owners[tuple(colors)] == family_id:
                        choices.append(tuple(colors))
            if not choices:
                raise ValueError("二维 donor 图像池不足，不能跨族复用")
            random.Random(seed + index * 97 + offset).shuffle(choices)
            colors = choices[0]
            owners[colors] = family_id
            donor_objects = tuple(replace(obj, color=colors[i]) for i, obj in enumerate(objects))
            pair_id = f"{family_id}-grid2-refmove-{relation}"
            recipient = _input(
                scene,
                objects,
                input_id=pair_id + "-clean",
                setting="squares_grid2",
                relation=relation,
                target_index=target,
                palette="paper",
                size=336,
                reference_index=reference_index,
            )
            donor = _input(
                scene,
                donor_objects,
                input_id=pair_id + "-donor",
                setting="squares_grid2",
                relation=relation,
                target_index=donor_target,
                palette="paper",
                size=336,
                reference_index=donor_reference,
            )
            prompt = recipient.prompt.replace(
                " " + PREPOSITIONS[relation] + " ",
                " immediately " + PREPOSITIONS[relation] + " ",
                1,
            )
            recipient = replace(recipient, prompt=prompt)
            donor = replace(donor, prompt=prompt)
            labels = (recipient.answer, b, donor.answer)
            if len(set(labels)) != 3 or donor.reference != recipient.reference:
                raise ValueError("二维控制必须同指令、同参照身份且三标签不同")
            result.append(
                PatchingPair(
                    pair_id,
                    family_id,
                    split,
                    "grid2_reference_move_without_reference_label_overlap",
                    recipient,
                    donor,
                    *labels,
                )
            )
    return result
