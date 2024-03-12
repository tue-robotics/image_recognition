# Yolo pose keypoint labels
# 0: nose
# 1: left-eye
# 2: right-eye
# 3: left-ear
# 4: right-ear
# 5: left-shoulder
# 6: right-shoulder
# 7: left-elbow
# 8: right-elbow
# 9: left-wrist
# 10: right-wrist
# 11: left-hip
# 12: right-hip
# 13: left-knee
# 14: right-knee
# 15: left-ankle
# 16: right-ankle

BODY_PARTS = {
    "Nose": "Nose",
    "LEye": "LEye",
    "REye": "REye",
    "LEar": "LEar",
    "REar": "REar",
    "LShoulder": "LShoulder",
    "RShoulder": "RShoulder",
    "LElbow": "LElbow",
    "RElbow": "RElbow",
    "LWrist": "LWrist",
    "RWrist": "RWrist",
    "LHip": "LHip",
    "RHip": "RHip",
    "LKnee": "LKnee",
    "RKnee": "RKnee",
    "LAnkle": "LAnkle",
    "RAnkle": "RAnkle",
}

BODY_PART_LINKS = [
    # The lowest index first
    # Matches the keys of BODY_PARTS
    # HEAD
    ("Nose", "LEye"),
    ("LEye", "LEar"),
    ("Nose", "REye"),
    ("LEye", "REye"),
    ("REye", "REar"),
    # Left side
    ("LEar", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("LShoulder", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),

    # Right side
    ("REar", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("RShoulder", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
]
