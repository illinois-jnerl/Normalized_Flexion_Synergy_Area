
# EMG-Based NFSA Analysis for Stroke Rehabilitation

This project contains a Python script for analyzing surface EMG signals and quantifying  
**Normalized Flexion Synergy Area (NFSA)** in stroke rehabilitation research.

---

## ✅ Key Features

- Bandpass filter (20–450 Hz) and envelope extraction via rectification + low-pass filter  
- Normalization to %MVC based on pull-in trial  
- Exponential curve fitting between deltoid and biceps activity  
- NFSA calculated as area under the curve between 10–90% of SABD  
- Average slope (dy/dx) of the fitted curve also calculated in the same range  
- Scatter and fitted plot visualization  

---

## 📁 Folder Structure

```
project_root/
├── emg_analysis_with_slope.py     # Main analysis script
├── README.md                      # Documentation (this file)
└── data/
    ├── subject_001/
    │   └── processed_data/
    │       ├── subject_001_liftup.csv
    │       └── subject_001_pullin_result.csv
    └── subject_002/
        └── processed_data/
```

---

## 📄 Required CSV Format

- `liftup.csv`: Contains columns `Trigger`, `Deltoid`, `Biceps`  
- `pullin_result.csv`: Single row where the **second column** is the MVC of biceps

---

## 🚀 How to Run

1. Install requirements:

```bash
pip install numpy pandas scipy matplotlib
```

2. Update `base_dir` and `processed_folder` in `emg_analysis_with_slope.py`

3. Run the script:

```bash
python emg_analysis_with_slope.py
```

---

## 📊 Output

- R² of exponential fit  
- NFSA % (area between x = 10–90)  
- Average slope (dy/dx) of exponential curve between x = 10 and x = 90  
- Plots showing exponential fit and shaded NFSA area

---

## 🔍 Interpretation

- **NFSA (%)**: Indicates the extent of synergistic biceps activation during shoulder abduction.  
- **Average Slope**: Reflects the rate of increase in synergistic biceps activation with increasing deltoid activity.

----------------------------------------------------------------
Please contact Dr. Yuan Yang (yuany@illinois.edu) for more information.

The code is associated with our paper:

[1] Sinha, N., Dewald, J. P. A., & Yang, Y. (2024). Perturbation-induced electromyographic activity is predictive of flexion synergy expression and a sensitive measure of post-stroke motor impairment. 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 1–4.

[2] Sung, J., Rajabtabar, M., Mulyana, B., Peng, H.-T., Yang, Y. (2024). The Expression of Flexion Synergy Enhances Spasticity in Stroke. 2024 SOCIETY for NEUROSCIENCE (SfN).
