# Project Name: CUPED Simulator

### **1. Executive Summary**

**CUPED Simulator** is a specialized simulation and planning tool designed to demonstrate the efficacy of **CUPED (Controlled-Experiment Using Pre-Experiment Data)** in industrial A/B testing.

While standard power analyses focus on *sample size*, this tool focuses on **variance reduction**. It quantifies exactly how much engineering time can be saved by utilizing pre-experiment covariates, helping teams move from "running tests for 4 weeks" to "running tests for 2 weeks" without sacrificing statistical rigor.

---

### **2. Problem Statement**

The #1 bottleneck in Product Experimentation is **Velocity**.

* **The Constraint:** High-variance metrics (e.g., Revenue per User) require massive sample sizes to detect small effects.
* **The Cost:** This forces experiments to run for weeks or months, exposing the business to "Cookie Churn" and delaying roadmap decisions.
* **The Solution:** Most teams ignore pre-experiment data. By using CUPED to remove explainable variance, we can mathematically "buy back" time.

---

### **3. User Personas**

* **The Experimentation Lead:** Needs to justify to stakeholders why investing in a complex method like CUPED is worth the engineering effort (ROI calculation).
* **The Data Engineer:** Needs to verify that the proposed statistical adjustments can scale to billions of rows without crashing the pipeline.

---

### **4. Functional Specifications**

#### **Module A: The Intuition Builder (Simulation Mode)**

*Target: Educational / Small Data Visualization*

* **Goal:** Visually demonstrate *how* regression adjustment shrinks error bars.
* **Inputs:**
* `Sample Size (N)`: Slider (e.g., 1,000 to 10,000).
* `Correlation (ρ)`: Slider (0.0 to 0.99) representing the strength of the pre-experiment covariate.
* `Treatment Effect`: (Optional) To show detection capabilities.


* **Backend Logic:**
* Generates synthetic data: .
* Calculates standard Difference-in-Means .
* Calculates CUPED-adjusted Mean .


* **Outputs (Visuals):**
* **The Shrinking Bell Curve:** Two overlapping histograms (Raw Metric vs. CUPED Metric) showing the variance reduction.
* **The "Noise" Reduction:** "Standard Error reduced from  to ."



#### **Module B: The Production Planner (Scale Mode)**

*Target: Real-world Decision Support*

* **Goal:** Calculate exact time savings for massive experiments using Sufficient Statistics (O(1) complexity).
* **Inputs:**
* `Baseline Daily Traffic`: (e.g., 500,000 users).
* `Baseline Metric Variance`: (From Data Warehouse).
* `Covariate Correlation (ρ)`: (From Data Warehouse).


* **Backend Logic:**
* Uses analytical variance reduction formula: .
* Calculates Sample Size Reduction Factor: .


* **Outputs (ROI):**
* **Efficiency Gain:** "Variance reduced by **X%**."
* **Time Savings:**
* "Standard Duration: **28 Days**"
* "CUPED Duration: **16 Days**"
* "**Result: You save 12 Days of testing time.**"

---

### **5. Technical Architecture**

#### **Frontend**

* **Framework:** Streamlit (Python) for rapid, interactive dashboarding.
* **Visualization:** `plotly.graph_objects` for dynamic distribution plots.

#### **Backend (Python)**

* **Simulation Engine:** `numpy` for vectorised random number generation.
* **Stats Engine:** `scipy.stats` for linear regression and variance calculations.
* **Architecture Pattern:** **Decoupled Compute**. The application logic is separated from data processing to ensure the tool remains performant regardless of dataset size.

#### **Scalability Validation (Offline)**

* **Component:** `validation_scripts/scale_test.py`
* **Technology:** **PySpark (Local Mode)**.
* **Purpose:** A standalone script included in the repo that generates 10M+ rows of synthetic data and performs the CUPED adjustment using Spark's vectorised operations.
* **Artifact:** A recorded technical demo linked in the README proving the method works on "Big Data."

---

### **7. Success Metrics**

1. **Visual Clarity:** Can a non-technical user look at the histogram and immediately understand "narrower is better"?
2. **Accuracy:** Does the simulation mode converge to the analytical formula results as  increases?
3. **Scalability Proof:** Does the repository contain the PySpark script proving O(N) scaling for data processing?


Todo:
- Create the data simulation function
- Replicate generating data, and generating a point estimate N times
- Generate a sampling distribution based off this
