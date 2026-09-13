def get_alert_modality(disability_type: str) -> dict:
    # Maps occupant profile -> alert modality
    mapping = {
        "deaf": {"visual": True, "vibration": True, "audio": False},
        "blind": {"visual": False, "vibration": True, "audio": True},
        "mobility": {"visual": True, "vibration": False, "audio": True},
        "cognitive": {"visual": True, "vibration": False, "audio": True},
        "none": {"visual": True, "vibration": False, "audio": True}
    }
    return mapping.get(disability_type.lower(), mapping["none"])
