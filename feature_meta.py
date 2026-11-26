"""Static clinical metadata for the 21 CTG features.

Display labels, one-sentence clinical descriptions, units, and the three feature
groups the frontend uses to organise its form. Numeric bounds (min/max/median)
are NOT here -- those are computed programmatically from the training set in
artifacts.py so they always reflect the data the models were trained on.
"""

# The three groups used to organise the input form.
GROUP_BASELINE = "Baseline & Movement"
GROUP_VARIABILITY = "Decelerations & Variability"
GROUP_HISTOGRAM = "FHR Histogram"

# feature name (exact CSV column) -> metadata
FEATURE_META = {
    "baseline value": {
        "label": "Baseline FHR",
        "description": "Baseline fetal heart rate, the average heart rate between contractions.",
        "unit": "bpm",
        "group": GROUP_BASELINE,
    },
    "accelerations": {
        "label": "Accelerations",
        "description": "Number of transient increases in fetal heart rate per second, a reassuring sign.",
        "unit": "per second",
        "group": GROUP_BASELINE,
    },
    "fetal_movement": {
        "label": "Fetal Movements",
        "description": "Number of fetal movements detected per second.",
        "unit": "per second",
        "group": GROUP_BASELINE,
    },
    "uterine_contractions": {
        "label": "Uterine Contractions",
        "description": "Number of uterine contractions per second.",
        "unit": "per second",
        "group": GROUP_BASELINE,
    },
    "light_decelerations": {
        "label": "Light Decelerations",
        "description": "Number of light (mild) fetal heart rate decelerations per second.",
        "unit": "per second",
        "group": GROUP_VARIABILITY,
    },
    "severe_decelerations": {
        "label": "Severe Decelerations",
        "description": "Number of severe fetal heart rate decelerations per second, a concerning sign.",
        "unit": "per second",
        "group": GROUP_VARIABILITY,
    },
    "prolongued_decelerations": {
        "label": "Prolonged Decelerations",
        "description": "Number of prolonged fetal heart rate decelerations per second, strongly associated with distress.",
        "unit": "per second",
        "group": GROUP_VARIABILITY,
    },
    "abnormal_short_term_variability": {
        "label": "Abnormal Short-Term Variability",
        "description": "Percentage of the recording with abnormal short-term beat-to-beat variability.",
        "unit": "%",
        "group": GROUP_VARIABILITY,
    },
    "mean_value_of_short_term_variability": {
        "label": "Mean Short-Term Variability",
        "description": "Mean value of short-term beat-to-beat heart rate variability.",
        "unit": "ms",
        "group": GROUP_VARIABILITY,
    },
    "percentage_of_time_with_abnormal_long_term_variability": {
        "label": "Abnormal Long-Term Variability",
        "description": "Percentage of the recording with abnormal long-term heart rate variability.",
        "unit": "%",
        "group": GROUP_VARIABILITY,
    },
    "mean_value_of_long_term_variability": {
        "label": "Mean Long-Term Variability",
        "description": "Mean value of long-term heart rate variability.",
        "unit": "ms",
        "group": GROUP_VARIABILITY,
    },
    "histogram_width": {
        "label": "Histogram Width",
        "description": "Width of the fetal heart rate histogram (max minus min).",
        "unit": "bpm",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_min": {
        "label": "Histogram Minimum",
        "description": "Minimum (lowest frequency) value of the fetal heart rate histogram.",
        "unit": "bpm",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_max": {
        "label": "Histogram Maximum",
        "description": "Maximum (highest frequency) value of the fetal heart rate histogram.",
        "unit": "bpm",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_number_of_peaks": {
        "label": "Histogram Peaks",
        "description": "Number of peaks in the fetal heart rate histogram.",
        "unit": "count",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_number_of_zeroes": {
        "label": "Histogram Zeroes",
        "description": "Number of zero-frequency bins in the fetal heart rate histogram.",
        "unit": "count",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_mode": {
        "label": "Histogram Mode",
        "description": "Most frequent fetal heart rate value in the histogram.",
        "unit": "bpm",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_mean": {
        "label": "Histogram Mean",
        "description": "Mean fetal heart rate value of the histogram.",
        "unit": "bpm",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_median": {
        "label": "Histogram Median",
        "description": "Median fetal heart rate value of the histogram.",
        "unit": "bpm",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_variance": {
        "label": "Histogram Variance",
        "description": "Variance of the fetal heart rate histogram, reflecting spread.",
        "unit": "bpm^2",
        "group": GROUP_HISTOGRAM,
    },
    "histogram_tendency": {
        "label": "Histogram Tendency",
        "description": "Symmetry tendency of the histogram (-1 left, 0 symmetric, 1 right).",
        "unit": "category",
        "group": GROUP_HISTOGRAM,
    },
}

GROUP_ORDER = [GROUP_BASELINE, GROUP_VARIABILITY, GROUP_HISTOGRAM]


def safe_name(feature: str) -> str:
    """Python-safe attribute name for a feature (only 'baseline value' needs it)."""
    return feature.replace(" ", "_")
