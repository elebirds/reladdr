"""生成正式 E01 的分组场景、直接相邻真值和保持参照有效的成对视觉反事实。"""

import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import permutations

from PIL import Image

from reladdr.tasks.synthetic import PALETTE, SHAPES, SceneObject, render_objects

GENERATOR_VERSION = "relational-grid-v1"
LAYOUTS = {
    "horizontal": (1, 3),
    "vertical": (3, 1),
    "grid2": (2, 2),
    "grid3": (3, 3),
    "grid4": (4, 4),
}
DIRECTIONS = {"left": (0, -1), "right": (0, 1), "above": (-1, 0), "below": (1, 0)}
CONDITIONS = ("original", "blank", "target_recolor", "position_swap")
TEMPLATES = ("relation_first", "reference_first")
SPLITS = ("discovery", "validation", "test")


@dataclass(frozen=True)
class RelationalScene:
    scene_id: str
    family_id: str
    split: str
    layout: str
    rows: int
    columns: int
    objects: tuple[SceneObject, ...]
    size: int = 448
    translation: tuple[int, int] = (0, 0)

    def record(self) -> dict:
        return asdict(self)

    def cell(self, row: int, column: int) -> SceneObject:
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise ValueError("查询超出网格边界")
        return self.objects[row * self.columns + column]

    def index(self, object_id: str) -> int:
        return next(i for i, obj in enumerate(self.objects) if obj.object_id == object_id)


@dataclass(frozen=True)
class RelationalQuestion:
    question_id: str
    pair_id: str
    kind: str
    template: str
    prompt: str
    answer: str
    foil: str
    target_id: str
    target_cell: tuple[int, int]
    reference_id: str | None = None
    reference_color: str | None = None
    reference_shape: str | None = None
    relation: str | None = None
    combination_partition: str = "not_applicable"
    input_view: str = "full_scene"

    def record(self) -> dict:
        return asdict(self)


def _centers(count: int) -> tuple[int, ...]:
    # 标准中心按 28 像素对齐；对象边长 56，不随密度偷偷缩小。
    return {1: (224,), 2: (140, 308), 3: (84, 224, 364), 4: (56, 168, 280, 392)}[count]


def make_grouped_scenes(seed: int, family_counts: dict[str, int], layouts: list[str]):
    if set(family_counts) != set(SPLITS) or any(
        type(n) is not int or n < 1 for n in family_counts.values()
    ):
        raise ValueError("必须为 discovery/validation/test 分别指定正整数场景族数量")
    if not layouts or len(layouts) != len(set(layouts)) or set(layouts) - LAYOUTS.keys():
        raise ValueError("布局须为不重复的已知布局")
    # 三对象锚点的形状与颜色均唯一，避免在生成阶段制造指代歧义。
    anchors = [tuple(zip(SHAPES, colors, strict=True)) for colors in permutations(PALETTE, 3)]
    if sum(family_counts.values()) > len(anchors):
        raise ValueError("本协议最多提供 120 个不同三对象属性锚点；扩量须修订生成协议")
    rng = random.Random(seed)
    rng.shuffle(anchors)
    # 预先固定整个有限锚点池的 split；增大某个 split 的数量不会重分已有场景族。
    assignments = ["discovery"] * 72 + ["validation"] * 24 + ["test"] * 24
    random.Random(seed + 913).shuffle(assignments)
    available = Counter(assignments)
    if any(family_counts[split] > available[split] for split in SPLITS):
        raise ValueError("本协议的稳定锚点池上限为 discovery=72、validation=24、test=24")
    selected = Counter()
    scenes = []
    for family_index, split in enumerate(assignments):
        if selected[split] >= family_counts[split]:
            continue
        selected[split] += 1
        family_rng = random.Random(seed * 1000 + family_index)
        anchor = list(anchors[family_index])
        family_rng.shuffle(anchor)
        remaining = [
            (shape, color) for shape in SHAPES for color in PALETTE if (shape, color) not in anchor
        ]
        family_rng.shuffle(remaining)
        inventory = anchor + remaining
        family_id = f"s{seed}-family-{family_index:04d}"
        # 每族共同平移不超过 5 像素，保留规则邻接；族间不复用完全相同的反事实图像。
        dx, dy = family_index % 11 - 5, family_index // 11 - 5
        for layout in layouts:
            rows, columns = LAYOUTS[layout]
            objects = []
            for i, (shape, color) in enumerate(inventory[: rows * columns]):
                row, column = divmod(i, columns)
                x, y = _centers(columns)[column] + dx, _centers(rows)[row] + dy
                objects.append(
                    SceneObject(f"object-{i:02d}", shape, color, (x - 28, y - 28, x + 28, y + 28))
                )
            scenes.append(
                RelationalScene(
                    f"{family_id}-{layout}",
                    family_id,
                    split,
                    layout,
                    rows,
                    columns,
                    tuple(objects),
                    translation=(dx, dy),
                )
            )
    validate_scenes(scenes)
    return scenes


def validate_scenes(scenes: list[RelationalScene]) -> None:
    family_splits = {}
    visible_scenes = {}
    for scene in scenes:
        previous = family_splits.setdefault(scene.family_id, scene.split)
        if previous != scene.split:
            raise ValueError("同一场景族不能跨数据划分")
        if len(scene.objects) != scene.rows * scene.columns:
            raise ValueError("主任务必须是无空位网格")
        if len({(obj.color, obj.shape) for obj in scene.objects}) != len(scene.objects):
            raise ValueError("颜色与形状组合必须唯一，以保证参照可识别")
        signature = tuple((obj.color, obj.shape, obj.bbox) for obj in scene.objects)
        # 使用可直接检查的对象清单，不用内容摘要替代数据验证。
        if signature in visible_scenes and visible_scenes[signature] != scene.family_id:
            raise ValueError("两个场景族生成了相同的完整原图")
        visible_scenes[signature] = scene.family_id


def combination_partitions(layout: str, seed: int) -> dict[tuple[int, int, str], str]:
    rows, columns = LAYOUTS[layout]
    result = {}
    for offset, (direction, (dr, dc)) in enumerate(DIRECTIONS.items()):
        references = [
            (r, c)
            for r in range(rows)
            for c in range(columns)
            if 0 <= r + dr < rows and 0 <= c + dc < columns
        ]
        # 1D 和 2×2 是共享复现锚点；3×3/4×4 才有足够位置做逐方向组合留出。
        if min(rows, columns) < 3:
            result.update({(r, c, direction): "shared" for r, c in references})
            continue
        random.Random(seed + rows * 100 + columns * 10 + offset).shuffle(references)
        heldout = max(1, len(references) // 6)
        for i, (row, column) in enumerate(references):
            partition = (
                "validation_only"
                if i < heldout
                else "test_only"
                if i < 2 * heldout
                else "discovery"
            )
            result[row, column, direction] = partition
    return result


def questions_for(scene: RelationalScene, *, templates: list[str], combination_seed: int):
    partitions = combination_partitions(scene.layout, combination_seed)
    allowed = {"shared", "discovery", f"{scene.split}_only"}
    questions = []
    for i, obj in enumerate(scene.objects):
        row, column = divmod(i, scene.columns)
        foil = next(other.color for other in scene.objects if other.color != obj.color)
        for kind in ("recognition", "absolute"):
            if kind == "recognition":
                prompt = "What color is the object? Answer with one color word."
            elif scene.layout in {"horizontal", "vertical"}:
                words = (
                    ("leftmost", "middle", "rightmost")
                    if scene.rows == 1
                    else ("topmost", "middle", "bottommost")
                )
                prompt = f"What color is the {words[i]} object? Answer with one color word."
            else:
                prompt = (
                    f"What color is the object in row {row + 1}, column {column + 1} "
                    "of the grid? Count rows from top to bottom and columns from left "
                    "to right. Answer with one color word."
                )
            pair_id = f"{scene.scene_id}-{kind}-{i:02d}"
            questions.append(
                RelationalQuestion(
                    pair_id,
                    pair_id,
                    kind,
                    "control",
                    prompt,
                    obj.color,
                    foil,
                    obj.object_id,
                    (row, column),
                    input_view="oracle_crop" if kind == "recognition" else "full_scene",
                )
            )
        for direction, (dr, dc) in DIRECTIONS.items():
            target_row, target_column = row + dr, column + dc
            if not (0 <= target_row < scene.rows and 0 <= target_column < scene.columns):
                continue
            partition = partitions[row, column, direction]
            if partition not in allowed:
                continue
            target = scene.cell(target_row, target_column)
            foil = (
                obj.color
                if obj.color != target.color
                else next(other.color for other in scene.objects if other.color != target.color)
            )
            preposition = f"to the {direction} of" if direction in {"left", "right"} else direction
            relative = (
                f"to its {direction}" if direction in {"left", "right"} else f"{direction} it"
            )
            pair_id = f"{scene.scene_id}-relation-{i:02d}-{direction}"
            for template in templates:
                if template == "relation_first":
                    prompt = (
                        f"The color of the shape immediately {preposition} the "
                        f"{obj.color} {obj.shape} is"
                    )
                elif template == "reference_first":
                    prompt = (
                        f"Look at the {obj.color} {obj.shape}. What color is the object "
                        f"immediately {relative}? Answer with one color word."
                    )
                else:
                    raise ValueError("未知问题模板")
                questions.append(
                    RelationalQuestion(
                        f"{pair_id}-{template}",
                        pair_id,
                        "relation",
                        template,
                        prompt,
                        target.color,
                        foil,
                        target.object_id,
                        (target_row, target_column),
                        obj.object_id,
                        obj.color,
                        obj.shape,
                        direction,
                        partition,
                    )
                )
    return questions


def displayed_target(scene: RelationalScene, question: RelationalQuestion) -> SceneObject:
    if question.kind != "relation":
        return scene.cell(*question.target_cell)
    matches = [
        i
        for i, obj in enumerate(scene.objects)
        if obj.color == question.reference_color and obj.shape == question.reference_shape
    ]
    if len(matches) != 1:
        raise ValueError("实际图像中参照描述不唯一或不存在")
    row, column = divmod(matches[0], scene.columns)
    dr, dc = DIRECTIONS[question.relation]
    return scene.cell(row + dr, column + dc)


def visual_condition(scene: RelationalScene, question: RelationalQuestion, condition: str):
    """生成本问题的实际图像；视觉反事实保留参照描述及其位置，重新从图像真值求答案。"""
    if condition not in CONDITIONS:
        raise ValueError("未知视觉条件")
    altered = scene
    changes = []
    target_index = scene.index(question.target_id)
    if condition == "target_recolor":
        objects = list(scene.objects)
        target = objects[target_index]
        used = {obj.color for obj in objects if obj.shape == target.shape}
        available = [color for color in PALETTE if color not in used]
        if available:
            objects[target_index] = replace(target, color=available[0])
            changes = [target.object_id]
        else:
            # 六色都被同形对象占用时交换两个颜色；参照本身保持不变，组合仍然唯一。
            donor_index = next(
                i
                for i, obj in enumerate(objects)
                if obj.shape == target.shape
                and obj.object_id not in {question.target_id, question.reference_id}
            )
            donor = objects[donor_index]
            objects[target_index] = replace(target, color=donor.color)
            objects[donor_index] = replace(donor, color=target.color)
            changes = [target.object_id, donor.object_id]
        altered = replace(scene, objects=tuple(objects))
    elif condition == "position_swap":
        objects = list(scene.objects)
        target = objects[target_index]
        donor_index = next(
            i
            for i, obj in enumerate(objects)
            if obj.object_id not in {question.target_id, question.reference_id}
            and obj.color != target.color
        )
        donor = objects[donor_index]
        objects[target_index] = replace(donor, bbox=target.bbox)
        objects[donor_index] = replace(target, bbox=donor.bbox)
        changes = [target.object_id, donor.object_id]
        altered = replace(scene, objects=tuple(objects))
    target = displayed_target(altered, question)
    image = render_objects(altered.objects, altered.size)
    crop_box = None
    if question.input_view == "oracle_crop":
        x0, y0, x1, y1 = scene.cell(*question.target_cell).bbox
        crop_box = (x0 - 28, y0 - 28, x1 + 28, y1 + 28)
        clipped = (
            max(0, crop_box[0]),
            max(0, crop_box[1]),
            min(scene.size, crop_box[2]),
            min(scene.size, crop_box[3]),
        )
        # 边缘裁剪用白色补齐，防止黑边或宽高变化成为识别对照的额外因素。
        crop = Image.new("RGB", (112, 112), "white")
        crop.paste(image.crop(clipped), (clipped[0] - crop_box[0], clipped[1] - crop_box[1]))
        image = crop
    if condition == "blank":
        image = Image.new("RGB", image.size, "white")
    details = {
        "displayed_answer": None if condition == "blank" else target.color,
        "displayed_target_id": None if condition == "blank" else target.object_id,
        "changed_object_ids": changes,
        "crop_box": crop_box,
        "displayed_scene": altered.record() if condition != "blank" else None,
    }
    return image, details


def dataset_counts(scenes, *, templates, combination_seed):
    counts = Counter()
    for scene in scenes:
        for question in questions_for(
            scene, templates=templates, combination_seed=combination_seed
        ):
            counts[scene.split, scene.layout, question.kind, question.template] += 1
    return [
        {"split": split, "layout": layout, "kind": kind, "template": template, "n": n}
        for (split, layout, kind, template), n in sorted(counts.items())
    ]
