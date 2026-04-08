from typing import Dict, Tuple

from fvgvisionai.processor.ultralytics_classes import PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK, ELECTRIC_SCOOTER


class ModelCategory:
    def __init__(self, label: str, color: Tuple[int, int, int, float], icon: str, model_class):
        self.label = label
        self.color = color
        self.icon = icon
        self.model_class = model_class

    def __str__(self) -> str:
        return f"{self.label.upper()}"


categories_dict: Dict[int, ModelCategory] = {
    PERSON: ModelCategory("person", (209, 209, 0, 0.5), "person", PERSON),
    BICYCLE: ModelCategory("bicycle", (47, 139, 237, 0.5), "bicycle", BICYCLE),
    CAR: ModelCategory("car", (42, 237, 139, 0.5), "car", CAR),
    MOTORCYCLE: ModelCategory("motorcycle", (56, 0, 255, 0.5), "motorcycle", MOTORCYCLE),
    BUS: ModelCategory("bus", (169, 10, 150, 0.5), "bus", BUS),
    TRUCK: ModelCategory("truck", (169, 255, 143, 0.5), "truck", TRUCK),
    ELECTRIC_SCOOTER: ModelCategory("electric_scooter", (209, 255, 143, 0.5), "scooter", ELECTRIC_SCOOTER)
}

all_category_list = [categories_dict[PERSON], categories_dict[BICYCLE],
                     categories_dict[CAR], categories_dict[MOTORCYCLE],
                     categories_dict[BUS], categories_dict[TRUCK],
                     categories_dict[ELECTRIC_SCOOTER]]
