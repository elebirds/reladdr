"""生成带精确对象、关系真值与场景族标识的三形状调试任务。"""

import random
from dataclasses import asdict, dataclass
from typing import Literal

from PIL import Image, ImageDraw

PALETTE = {
    "red": (220, 40, 40),
    "green": (20, 150, 60),
    "blue": (35, 75, 220),
    "yellow": (240, 210, 25),
    "purple": (145, 50, 185),
    "orange": (240, 125, 20),
}
SHAPES = ("square", "circle", "triangle")
GENERATOR_VERSION = "three-shapes-v1"


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    shape: str
    color: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Scene:
    scene_id: str
    family_id: str
    orientation: Literal["horizontal", "vertical"]
    objects: tuple[SceneObject, ...]
    size: int = 448

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Question:
    question_id: str
    kind: Literal["recognition", "absolute", "relation"]
    prompt: str
    answer: str
    target_id: str
    foil: str
    reference_id: str | None = None
    relation: str | None = None

    def record(self) -> dict:
        return asdict(self)


def make_scenes(seed: int, families: int) -> list[Scene]:
    if type(families) is not int or families < 1:
        raise ValueError("场景族数量必须是正整数")
    rng = random.Random(seed)
    scenes = []
    for family in range(families):
        colors = rng.sample(list(PALETTE), 3)
        shapes = rng.sample(list(SHAPES), 3)
        family_id = f"family-{family:04d}"
        # 同一族的横/竖布局共享属性，不允许在后续正式数据划分时拆散。
        for orientation in ("horizontal", "vertical"):
            objects = []
            for index, (shape, color) in enumerate(zip(shapes, colors, strict=True)):
                x, y = (
                    (56 + 140 * index, 196)
                    if orientation == "horizontal"
                    else (196, 56 + 140 * index)
                )
                objects.append(SceneObject(f"object-{index}", shape, color, (x, y, x + 56, y + 56)))
            scenes.append(
                Scene(f"{family_id}-{orientation}", family_id, orientation, tuple(objects))
            )
    return scenes


def render_scene(scene: Scene) -> Image.Image:
    image = Image.new("RGB", (scene.size, scene.size), "white")
    draw = ImageDraw.Draw(image)
    for obj in scene.objects:
        x0, y0, x1, y1 = obj.bbox
        color = PALETTE[obj.color]
        if obj.shape == "square":
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)
        elif obj.shape == "circle":
            draw.ellipse((x0, y0, x1 - 1, y1 - 1), fill=color)
        elif obj.shape == "triangle":
            draw.polygon((((x0 + x1 - 1) // 2, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)), fill=color)
        else:
            raise ValueError("未知形状")
    return image


def questions_for(scene: Scene) -> list[Question]:
    questions = []
    positions = (
        ("leftmost", "middle", "rightmost")
        if scene.orientation == "horizontal"
        else ("topmost", "middle", "bottommost")
    )
    for index, obj in enumerate(scene.objects):
        questions.append(
            Question(
                f"{scene.scene_id}-recognition-{index}",
                "recognition",
                f"What color is the {obj.shape}? Answer with one color word.",
                obj.color,
                obj.object_id,
                scene.objects[(index + 1) % 3].color,
            )
        )
        questions.append(
            Question(
                f"{scene.scene_id}-absolute-{index}",
                "absolute",
                f"What color is the {positions[index]} object? Answer with one color word.",
                obj.color,
                obj.object_id,
                scene.objects[(index + 1) % 3].color,
            )
        )
    # 关系语义为同一行/列中的直接相邻对象；边界外不生成没有答案的问题。
    directions = (
        ((-1, "left", "to its left"), (1, "right", "to its right"))
        if scene.orientation == "horizontal"
        else ((-1, "above", "above it"), (1, "below", "below it"))
    )
    for reference_index, reference in enumerate(scene.objects):
        for delta, relation, wording in directions:
            target_index = reference_index + delta
            if not 0 <= target_index < len(scene.objects):
                continue
            target = scene.objects[target_index]
            questions.append(
                Question(
                    f"{scene.scene_id}-relation-{reference_index}-{relation}",
                    "relation",
                    f"Look at the {reference.shape}. "
                    f"What color is the object immediately {wording}? Answer with one color word.",
                    target.color,
                    target.object_id,
                    reference.color,
                    reference.object_id,
                    relation,
                )
            )
    return questions
