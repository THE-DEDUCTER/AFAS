from ml.routing import get_accessible_route

def get_alert_payload(is_fire: bool, severity: str, disability_type: str) -> dict:
    """
    Returns the alert configuration based on fire presence and occupant needs.
    """
    if not is_fire:
        return {
            "trigger": False,
            "visual": False,
            "vibration": False,
            "audio": False,
            "message": "Status Normal. No fire detected.",
            "guidance": None
        }

    disability_type = disability_type.lower()
    
    # Default (no disability) gets standard audio alarm
    alert_config = {
        "visual": False, 
        "vibration": False, 
        "audio": True
    }
    
    guidance = "Evacuate using the nearest stairwell."

    if disability_type == "deaf":
        alert_config["visual"] = True
        alert_config["vibration"] = True
        alert_config["audio"] = False
        guidance = "Follow visual strobes to the nearest exit."
    elif disability_type == "blind":
        alert_config["vibration"] = True
        alert_config["audio"] = True
        guidance = "Fire detected. Please follow the auditory beacons to the exit."
    elif disability_type == "mobility":
        alert_config["audio"] = True
        alert_config["visual"] = True
        
        # Call the routing engine stretch feature!
        safe_route = get_accessible_route("Room 101")
        guidance = f"Do not use stairs. Your accessible evacuation route: {safe_route}"
    elif disability_type == "cognitive":
        alert_config["audio"] = True
        alert_config["visual"] = True
        guidance = "Fire alarm. Please follow the green exit lights calmly."

    return {
        "trigger": True,
        "severity": severity,
        "visual": alert_config["visual"],
        "vibration": alert_config["vibration"],
        "audio": alert_config["audio"],
        "message": f"Fire Alert! Severity: {severity}.",
        "guidance": guidance
    }
