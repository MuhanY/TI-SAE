# tisae/models/ontology.py
"""
Clinical Ontology Mapping for SOFA-based Channel Assignment

Maps MIMIC-III variables to 10 clinically meaningful channels based on
SOFA (Sequential Organ Failure Assessment) organ systems, with multi-channel
support for variables that span multiple organ systems.

Channels:
0. Respiratory (PaO2, FiO2, SpO2, ventilation, blood gases)
1. Cardiovascular (MAP, vasopressors, HR, BP, IV fluids)
2. Coagulation (Platelets, INR, PT, PTT, FFP)
3. Liver (Bilirubin, liver enzymes, albumin, INR/PT)
4. Renal (Creatinine, BUN, urine output, diuretics, IV fluids)
5. Neurological (GCS, sedation, opioids)
6. Metabolic (Glucose, electrolytes, acid-base, blood gases)
7. Hematologic (Hgb, WBC, RBC indices, transfusions)
8. Infection/Inflammatory (Temperature, WBC, antibiotics)
9. Other/Miscellaneous (remaining variables)

Multi-channel variables (e.g. blood gases → Resp+Metabolic, Lactate →
Cardio+Metabolic) have a primary channel for routing and appear in
multiple channels for decoder masking via get_hard_mask().
"""

from __future__ import annotations
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ChannelDefinition:
    """Definition of a clinical channel."""
    idx: int
    name: str
    short_name: str
    description: str
    sofa_component: Optional[str]


# =============================================================================
# SOFA-based Channel Definitions (10 channels)
# =============================================================================

CHANNEL_DEFINITIONS = [
    ChannelDefinition(
        idx=0, name="Respiratory", short_name="resp",
        description="Respiratory function and oxygenation",
        sofa_component="Respiration (PaO2/FiO2)",
    ),
    ChannelDefinition(
        idx=1, name="Cardiovascular", short_name="cardio",
        description="Cardiovascular function and hemodynamics",
        sofa_component="Cardiovascular (MAP, vasopressors)",
    ),
    ChannelDefinition(
        idx=2, name="Coagulation", short_name="coag",
        description="Coagulation and clotting function",
        sofa_component="Coagulation (Platelets)",
    ),
    ChannelDefinition(
        idx=3, name="Liver", short_name="liver",
        description="Liver function and metabolism",
        sofa_component="Liver (Bilirubin)",
    ),
    ChannelDefinition(
        idx=4, name="Renal", short_name="renal",
        description="Renal function and fluid balance",
        sofa_component="Renal (Creatinine, Urine output)",
    ),
    ChannelDefinition(
        idx=5, name="Neurological", short_name="neuro",
        description="Neurological function and consciousness",
        sofa_component="CNS (Glasgow Coma Scale)",
    ),
    ChannelDefinition(
        idx=6, name="Metabolic", short_name="metab",
        description="Metabolic function and acid-base balance",
        sofa_component=None,
    ),
    ChannelDefinition(
        idx=7, name="Hematologic", short_name="heme",
        description="Hematologic indices (non-coagulation)",
        sofa_component=None,
    ),
    ChannelDefinition(
        idx=8, name="Infection", short_name="infect",
        description="Infection and inflammatory markers",
        sofa_component=None,
    ),
    ChannelDefinition(
        idx=9, name="Other", short_name="other",
        description="Miscellaneous and unclassified variables",
        sofa_component=None,
    ),
]


# =============================================================================
# Explicit Variable → Channel Mapping
#
# Format: "Exact Variable Name": [primary_channel, secondary_channel, ...]
# The first channel is used for routing (get_channel_assignment).
# All channels are used for decoder masking (get_hard_mask).
#
# Channel indices:
#   0=Respiratory  1=Cardiovascular  2=Coagulation  3=Liver  4=Renal
#   5=Neurological 6=Metabolic       7=Hematologic   8=Infection  9=Other
# =============================================================================

VARIABLE_CHANNEL_MAP: Dict[str, List[int]] = {
    # --- Ch0: Respiratory ---
    "O2 Saturation":     [0],
    "RR":                [0],
    "FiO2":              [0],
    "Intubated":         [0],
    "Chest Tube":        [0],
    "PO2":               [0, 6],
    "PCO2":              [0, 6],
    "pH Blood":          [0, 6],
    "Total CO2":         [0, 6],
    "Base Excess":       [0, 6],
    "Propofol":          [5, 0],
    "Fentanyl":          [5, 0],

    # --- Ch1: Cardiovascular ---
    "MBP":               [1],
    "HR":                [1],
    "SBP":               [1],
    "DBP":               [1],
    "Vasopressin":       [1],
    "Neosynephrine":     [1],
    "Nitroglycerine":    [1],
    "Amiodarone":        [1],
    "Norepinephrine":    [1],
    "Epinephrine":       [1],
    "Milrinone":         [1],
    "Nitroprusside":     [1],
    "Troponin T":        [1],
    "Troponin I":        [1],
    "NTproBNP":          [1],
    "Hydralazine":       [1],
    "Metoprolol":        [1],
    "CRR":               [1],
    "Lactate":           [6, 1],
    "Dopamine":          [1, 4],
    "Furosemide":        [4, 1],
    "Heparin":           [2, 1],
    "Normal Saline":     [1, 4],
    "Lactated Ringers":  [1, 4],
    "Colloid":           [1, 4],
    "OR/PACU Crystalloid": [1, 4],
    "Weight":            [1, 4],

    # --- Ch2: Coagulation ---
    "PTT":               [2],
    "Fibrinogen":        [2],
    "D-Dimer":           [2],
    "Platelet Count":    [2, 7],
    "Fresh Frozen Plasma": [2, 7],
    "INR":               [2, 3],
    "PT":                [2, 3],

    # --- Ch3: Liver ---
    "Bilirubin (Total)": [3],
    "Bilirubin (Direct)": [3],
    "Bilirubin (Indirect)": [3],
    "ALP":               [3],
    "ALT":               [3],
    "AST":               [3],
    "Albumin":           [3],
    "Ammonia":           [3],
    "LDH":               [3, 7],
    "Albumin 25%":       [3, 4],
    "Albumin 5%":        [3, 4],

    # --- Ch4: Renal ---
    "Creatinine Blood":  [4],
    "Creatinine Urine":  [4],
    "BUN":               [4],
    "Urine":             [4],
    "SG Urine":          [4],
    "pH Urine":          [4],
    "Ultrafiltrate":     [4],

    # --- Ch5: Neurological ---
    "GCS_eye":           [5],
    "GCS_verbal":        [5],
    "GCS_motor":         [5],
    "Midazolam":         [5],
    "Lorazepam":         [5],
    "Hydromorphone":     [5],
    "Morphine Sulfate":  [5],

    # --- Ch6: Metabolic ---
    "Anion Gap":         [6],
    "Glucose (Serum)":   [6],
    "Glucose (Whole Blood)": [6],
    "Glucose (Blood)":   [6],
    "Sodium":            [6],
    "Calcium Total":     [6],
    "Chloride":          [6],
    "Potassium":         [6],
    "Bicarbonate":       [6],
    "Phosphate":         [6],
    "Magnesium":         [6],
    "Calcium Free":      [6],
    "Calcium Gluconate": [6],
    "Magnesium Sulphate": [6],
    "Magnesium Sulfate (Bolus)": [6],
    "KCl":               [6],
    "KCl (Bolus)":       [6],
    "Insulin Regular":   [6],
    "Insulin Humalog":   [6],
    "Insulin largine":   [6],
    "Insulin NPH":       [6],

    # --- Ch7: Hematologic ---
    "Hgb":               [7],
    "Hct":               [7],
    "RBC":               [7],
    "MCV":               [7],
    "MCH":               [7],
    "MCHC":              [7],
    "RDW":               [7],
    "Packed RBC":        [7],
    "WBC":               [7, 8],

    # --- Ch8: Infection ---
    "Temperature":       [8],
    "Monocytes":         [8],
    "Neutrophils":       [8],
    "Basophils":         [8],
    "Lymphocytes":       [8],
    "Lymphocytes (Absolute)": [8],
    "Eoisinophils":      [8],
    "Cefazolin":         [8],
    "Vacomycin":         [8],
    "Piperacillin":      [8],
    "CRP":               [8],

    # --- Ch9: Other ---
    "Unknown":           [9],
    "Solution":          [9],
    "D5W":               [9],
    "Dextrose Other":    [9],
    "PO intake":         [9],
    "Gastric":           [9],
    "Gastric Meds":      [9],
    "GT Flush":          [9],
    "Sterile Water":     [9],
    "Stool":             [9],
    "Jackson-Pratt":     [9],
    "Fiber":             [9],
    "TPN":               [9],
    "Half Normal Saline": [9],
    "Pantoprazole":      [9],
    "Free Water":        [9],
    "Pre-admission Intake": [9],
    "Piggyback":         [9],
    "Famotidine":        [9],
    "Pre-admission Output": [9],
    "Height":            [9],
    "Emesis":            [9],
    "Residual":          [9],
    "EBL":               [9],
}


# =============================================================================
# Ontology Mapping Class
# =============================================================================

class ClinicalOntologyMapping:
    """
    Maps variable names/indices to clinical channels.

    Supports multi-channel assignment: a variable can belong to multiple
    channels (e.g. INR → Coagulation + Liver). The first channel in the
    list is the "primary" channel used for routing.

    Usage:
        ontology = ClinicalOntologyMapping(var_names)
        channel_assignment = ontology.get_channel_assignment()  # [V] primary channel
        mask = ontology.get_hard_mask()          # [C, V] multi-hot mask
    """

    def __init__(self, var_names: List[str], n_channels: int = 10):
        self.var_names = var_names
        self.V = len(var_names)
        self.n_channels = n_channels

        self.var_to_channels: Dict[int, List[int]] = self._assign_variables()
        self.var_to_channel: Dict[int, int] = {
            v: chs[0] for v, chs in self.var_to_channels.items()
        }
        self._compute_statistics()

    def _assign_variables(self) -> Dict[int, List[int]]:
        """Assign each variable index to one or more channels using exact lookup."""
        other_idx = self.n_channels - 1
        assignments: Dict[int, List[int]] = {}

        if self.n_channels == 1:
            for v_idx in range(self.V):
                assignments[v_idx] = [0]
            return assignments

        for v_idx, var_name in enumerate(self.var_names):
            if var_name in VARIABLE_CHANNEL_MAP:
                channels = VARIABLE_CHANNEL_MAP[var_name]
                assignments[v_idx] = [c for c in channels if c < self.n_channels]
                if not assignments[v_idx]:
                    assignments[v_idx] = [other_idx]
            else:
                assignments[v_idx] = [other_idx]

        return assignments

    def _compute_statistics(self):
        """Compute assignment statistics."""
        self.channel_counts = [0] * self.n_channels
        self.channel_variables: Dict[int, List[str]] = {i: [] for i in range(self.n_channels)}

        for v_idx, channels in self.var_to_channels.items():
            for c_idx in channels:
                self.channel_counts[c_idx] += 1
                self.channel_variables[c_idx].append(self.var_names[v_idx])

    def get_channel_assignment(self) -> torch.Tensor:
        """Get primary channel assignment tensor [V]."""
        assignment = torch.zeros(self.V, dtype=torch.long)
        for v_idx, c_idx in self.var_to_channel.items():
            assignment[v_idx] = c_idx
        return assignment

    def get_hard_mask(self) -> torch.Tensor:
        """Get hard mask matrix [C, V] for decoder variable restriction."""
        mask = torch.zeros(self.n_channels, self.V)
        for v_idx, channels in self.var_to_channels.items():
            for c_idx in channels:
                mask[c_idx, v_idx] = 1.0
        return mask

    def get_attention_bias(self, bias_strength: float = 10.0) -> torch.Tensor:
        """Get attention bias matrix [C, V] for soft ontology guidance."""
        bias = torch.zeros(self.n_channels, self.V)
        for v_idx, c_idx in self.var_to_channel.items():
            bias[c_idx, v_idx] = bias_strength
        return bias

    def get_multi_channel_variables(self) -> Dict[str, List[str]]:
        """Get variables assigned to multiple channels, with channel names."""
        result = {}
        for v_idx, channels in self.var_to_channels.items():
            if len(channels) > 1:
                names = [CHANNEL_DEFINITIONS[c].name for c in channels]
                result[self.var_names[v_idx]] = names
        return result

    def print_summary(self):
        """Print summary of channel assignments."""
        print("=" * 60)
        print("Clinical Ontology Channel Assignment Summary")
        print("=" * 60)

        for i, channel_def in enumerate(CHANNEL_DEFINITIONS[:self.n_channels]):
            count = self.channel_counts[i]
            pct = 100 * count / self.V
            print(f"\nChannel {i}: {channel_def.name} ({count} vars, {pct:.1f}%)")
            print(f"  SOFA: {channel_def.sofa_component or 'N/A'}")

            vars_in_channel = self.channel_variables[i]
            if vars_in_channel:
                display = []
                for vn in vars_in_channel:
                    v_idx = self.var_names.index(vn)
                    suffix = "*" if len(self.var_to_channels[v_idx]) > 1 else ""
                    display.append(vn + suffix)
                show = display[:10]
                if len(display) > 10:
                    show.append(f"... +{len(display) - 10} more")
                print(f"  Variables: {', '.join(show)}")

        multi = self.get_multi_channel_variables()
        if multi:
            print(f"\n* = multi-channel ({len(multi)} variables)")

        print("\n" + "=" * 60)

    def get_channel_name(self, channel_idx: int) -> str:
        """Get human-readable name for channel."""
        if 0 <= channel_idx < len(CHANNEL_DEFINITIONS):
            return CHANNEL_DEFINITIONS[channel_idx].name
        return f"Channel_{channel_idx}"

    def get_channel_names(self) -> List[str]:
        """Get list of all channel names."""
        return [self.get_channel_name(i) for i in range(self.n_channels)]


def create_ontology_mapping(var_names: List[str], n_channels: int = 10) -> ClinicalOntologyMapping:
    """Create ontology mapping from variable names."""
    return ClinicalOntologyMapping(var_names, n_channels)
