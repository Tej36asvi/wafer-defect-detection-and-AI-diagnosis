# defect_context.py
# Fab-grade Defect Knowledge Base
# This is used as the physics + process truth layer for the RAG system

DEFECT_KNOWLEDGE_BASE = {

    # -------------------------------
    # 1. Center Defect
    # -------------------------------
    "Center": {
        "Process_Module": "Lithography (Spin Coating / Develop)",
        "Physics": (
            "Non-uniform centrifugal spreading or solvent evaporation "
            "causing resist thickness or development anomalies at the wafer center."
        ),
        "Checklist": [
            "Verify spin chuck vacuum and centering",
            "Check photoresist dispense volume and nozzle condition",
            "Verify spin speed and acceleration profile",
            "Check bake plate center temperature uniformity"
        ],
        "Risk_Level": "Medium"
    },

    # -------------------------------
    # 2. Donut Defect
    # -------------------------------
    "Donut": {
        "Process_Module": "Thermal Processing (RTP / Furnace / Etch)",
        "Physics": (
            "Radial non-uniformity in temperature, plasma density, "
            "or gas flow producing annular reaction rate variation."
        ),
        "Checklist": [
            "Check RTP lamp zoning or furnace heater uniformity",
            "Inspect gas distribution showerhead",
            "Verify wafer backside helium cooling",
            "Review temperature or plasma radial profiles"
        ],
        "Risk_Level": "High"
    },

    # -------------------------------
    # 3. Edge-Loc Defect
    # -------------------------------
    "Edge-Loc": {
        "Process_Module": "Wafer Handling / Transport",
        "Physics": (
            "Mechanical contact, edge chipping, or stress concentration "
            "introduced during robotic handling or load/unload."
        ),
        "Checklist": [
            "Recalibrate robot arm and wafer centering",
            "Inspect end effector and edge grip pads",
            "Check cassette, FOUP, and aligner offsets"
        ],
        "Risk_Level": "Low"
    },

    # -------------------------------
    # 4. Edge-Ring Defect
    # -------------------------------
    "Edge-Ring": {
        "Process_Module": "Lithography (Spin Coat / Edge Bead Removal) or Wet Clean",
        "Physics": (
            "Residual edge bead or edge process exclusion zone "
            "causing a continuous ring of film or contamination."
        ),
        "Checklist": [
            "Verify edge bead removal (EBR) solvent flow",
            "Inspect clamp ring and wafer edge exposure",
            "Check spin recipe edge exclusion parameters"
        ],
        "Risk_Level": "Medium"
    },

    # -------------------------------
    # 5. Localized (Loc) Defect
    # -------------------------------
    "Loc": {
        "Process_Module": "Deposition / Etch / Implant",
        "Physics": (
            "Particle contamination or micro-mask originating from "
            "chamber walls, shields, or process residues."
        ),
        "Checklist": [
            "Run chamber particle monitor or wafer",
            "Inspect shields, liners, and focus rings",
            "Check recent chamber clean history",
            "Review vacuum pump and exhaust performance"
        ],
        "Risk_Level": "High"
    },

    # -------------------------------
    # 6. Random Defect
    # -------------------------------
    "Random": {
        "Process_Module": "Cleanroom Environment / Tool Cross-Contamination",
        "Physics": (
            "Stochastic particle deposition from airborne contamination, "
            "operator handling, or tool-to-tool cross contamination."
        ),
        "Checklist": [
            "Check HEPA filter performance",
            "Inspect FOUPs, SMIF pods, and load ports",
            "Verify ionizer and ESD control",
            "Review tool maintenance history"
        ],
        "Risk_Level": "Variable"
    },

    # -------------------------------
    # 7. Scratch
    # -------------------------------
    "Scratch": {
        "Process_Module": "CMP / Handling",
        "Physics": (
            "Mechanical abrasion from trapped slurry particles, pad debris, "
            "or misaligned wafer handling surfaces."
        ),
        "Checklist": [
            "Inspect CMP pad and conditioner disk",
            "Check slurry filtration and particle counts",
            "Inspect wafer handling tracks and rollers",
            "Review recent tool alarms"
        ],
        "Risk_Level": "Critical"
    },

    # -------------------------------
    # 8. Near-Full Defect
    # -------------------------------
    "Near-full": {
        "Process_Module": "Process Tool or Metrology Tool Failure",
        "Physics": (
            "Catastrophic process excursion or inspection tool saturation "
            "leading to wafer-wide apparent defects."
        ),
        "Checklist": [
            "Verify inspection tool calibration",
            "Cross-check with a second metrology tool",
            "Review process chamber alarms and logs",
            "Inspect wafer for actual film or etch failure"
        ],
        "Risk_Level": "Critical"
    },

    # -------------------------------
    # 9. No Defect
    # -------------------------------
    "none": {
        "Process_Module": "Normal Operation",
        "Physics": "Process within statistical control limits.",
        "Checklist": [
            "Release wafer to next process step"
        ],
        "Risk_Level": "None"
    }
}