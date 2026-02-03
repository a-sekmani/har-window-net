"""Label map utilities: NTU A001..A120 to id/name for cloud compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# NTU RGB+D 120 action labels A001..A120 (short names for label_map)
NTU120_ACTION_NAMES: dict[str, str] = {
    "A001": "drink water",
    "A002": "eat meal/snack",
    "A003": "brushing teeth",
    "A004": "brushing hair",
    "A005": "drop",
    "A006": "pickup",
    "A007": "throw",
    "A008": "sitting down",
    "A009": "standing up",
    "A010": "clapping",
    "A011": "reading",
    "A012": "writing",
    "A013": "tear up paper",
    "A014": "wear jacket",
    "A015": "take off jacket",
    "A016": "wear a shoe",
    "A017": "take off a shoe",
    "A018": "wear on glasses",
    "A019": "take off glasses",
    "A020": "put on a hat/cap",
    "A021": "take off a hat/cap",
    "A022": "cheer up",
    "A023": "hand waving",
    "A024": "kicking something",
    "A025": "reach into pocket",
    "A026": "hopping",
    "A027": "jump up",
    "A028": "make a phone call",
    "A029": "playing with phone/tablet",
    "A030": "typing on a keyboard",
    "A031": "pointing to something",
    "A032": "taking a selfie",
    "A033": "check time",
    "A034": "rub two hands together",
    "A035": "nod head/bow",
    "A036": "shake head",
    "A037": "wipe face",
    "A038": "salute",
    "A039": "put the palms together",
    "A040": "cross hands in front",
    "A041": "sneeze/cough",
    "A042": "staggering",
    "A043": "falling",
    "A044": "touch head",
    "A045": "touch chest",
    "A046": "touch back",
    "A047": "touch neck",
    "A048": "nausea or vomiting",
    "A049": "use a fan",
    "A050": "punching/slapping other person",
    "A051": "kicking other person",
    "A052": "pushing other person",
    "A053": "pat on back of other person",
    "A054": "point finger at the other person",
    "A055": "hugging other person",
    "A056": "giving something to other person",
    "A057": "touch other person's pocket",
    "A058": "handshaking",
    "A059": "walking towards each other",
    "A060": "walking apart from each other",
    "A061": "put on headphone",
    "A062": "take off headphone",
    "A063": "shoot at the basket",
    "A064": "bounce ball",
    "A065": "tennis bat swing",
    "A066": "juggling table tennis balls",
    "A067": "hush",
    "A068": "flick hair",
    "A069": "thumb up",
    "A070": "thumb down",
    "A071": "make ok sign",
    "A072": "make victory sign",
    "A073": "staple book",
    "A074": "counting money",
    "A075": "cutting nails",
    "A076": "cutting paper",
    "A077": "snapping fingers",
    "A078": "open bottle",
    "A079": "sniff (smell)",
    "A080": "squat down",
    "A081": "toss a coin",
    "A082": "fold paper",
    "A083": "ball up paper",
    "A084": "play magic cube",
    "A085": "apply cream on face",
    "A086": "apply cream on hand back",
    "A087": "put on bag",
    "A088": "take off bag",
    "A089": "put something into a bag",
    "A090": "take something out of a bag",
    "A091": "open a box",
    "A092": "move heavy objects",
    "A093": "shake fist",
    "A094": "throw up cap/hat",
    "A095": "hands up (both hands)",
    "A096": "cross arms",
    "A097": "arm circles",
    "A098": "arm swings",
    "A099": "running on the spot",
    "A100": "butt kicks",
    "A101": "cross toe touch",
    "A102": "side kick",
    "A103": "yawn",
    "A104": "stretch oneself",
    "A105": "blow nose",
    "A106": "hit other person with something",
    "A107": "wield knife towards other person",
    "A108": "knock over other person",
    "A109": "grab other person's stuff",
    "A110": "shoot at other person with a gun",
    "A111": "step on foot",
    "A112": "high-five",
    "A113": "cheers and drink",
    "A114": "carry something with other person",
    "A115": "take a photo of other person",
    "A116": "follow other person",
    "A117": "whisper in other person's ear",
    "A118": "exchange things with other person",
    "A119": "support somebody with hand",
    "A120": "finger-guessing game",
}


def build_default_label_map() -> dict[str, Any]:
    """Build label_map: id (int 0..119) -> name, and index_by_label A001 -> 0."""
    index_by_label: dict[str, int] = {}
    id_to_name: dict[str, str] = {}
    for i, (label, name) in enumerate(NTU120_ACTION_NAMES.items()):
        idx_str = str(i)
        id_to_name[idx_str] = name
        index_by_label[label] = i
    return {
        "id_to_name": id_to_name,
        "label_to_id": index_by_label,
        "num_classes": 120,
    }


def load_label_map(path: str | Path) -> dict[str, Any]:
    """Load label_map.json from path."""
    with open(path, "rb") as f:
        return json.loads(f.read())


def save_label_map(label_map: dict[str, Any], path: str | Path) -> None:
    """Save label_map to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(json.dumps(label_map, indent=2).encode("utf-8"))


def get_label_id(label_map: dict[str, Any], label: str) -> int:
    """Get integer id for NTU label (e.g. A001 -> 0)."""
    return label_map["label_to_id"][label]


def get_label_name(label_map: dict[str, Any], label_id: int) -> str:
    """Get action name for integer id."""
    return label_map["id_to_name"][str(label_id)]
