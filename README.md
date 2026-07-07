# Astraea: Autonomous Spacecraft AI Companion

Astraea is an interactive, out-of-this-world spacecraft command and simulation console. It fetches real-time solar storm, wind speed, and magnetic field data directly from NOAA space weather satellites and runs a high-performance orbital mechanics physics engine in Rust to predict spacecraft orbital decay, shielding deterioration, and radiation dose rates.

An AI companion (Astraea-9000) monitors this telemetry and guides the spacecraft through solar radiation storms, geomagnetic storms, and altitude degradation by writing autonomous log entries and suggesting corrective thruster maneuvers.

## Features

1. **Live Space Weather Integration**: Queries NOAA SWPC's live APIs for real-time solar wind speeds ($V_{sw}$), interplanetary magnetic field ($B_t$), and geomagnetic scale indicators ($G$, $S$, $R$).
2. **Rust Physics Engine**: Computes orbital drag using a piecewise barometric density model of LEO thermosphere adjusted for thermospheric heating during solar storms. Calculates radiation attenuation from satellite shielding.
3. **Interactive TUI Dashboard**: Features a gorgeous terminal dashboard built with Python `rich` featuring telemetry metrics, a live ASCII orbit visualizer, and the AI log interface.
4. **Agentic Reasoning (LLM / Fallback)**: Supports Gemini, OpenAI, or Hugging Face serverless APIs for generative ship companion logs, with a robust local rules-based engine as a zero-setup fallback.

---

## Installation & Setup

### Prerequisites
* Rust compiler (cargo 1.70+)
* Python (3.9+)

### 1. Build the Rust Physics Simulator
Compile the high-performance physics calculator:
```bash
cargo build --release
```

### 2. Configure Python Environment
Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)
Create a `.env` file in the project folder to enable advanced generative AI ship computer logs (if left blank, Astraea will automatically fall back to its local expert rules-based analyzer):
```env
GEMINI_API_KEY=your_gemini_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
# OR
HF_API_KEY=your_hugging_face_token_here
```

---

## Running the Dashboard

Launch the spacecraft console:
```bash
python agent.py
```

### Console Commands
* **[1] Boost Orbit (Low)**: Expends 10 kg of monopropellant to perform a prograde burn of +25 m/s delta-v, lifting the altitude.
* **[2] Boost Orbit (High)**: Expends 30 kg of monopropellant to perform a prograde burn of +75 m/s delta-v, significantly raising altitude.
* **[3] Deploy Radiation Shielding**: deploys active magnetic/physical shielding (+1.5 g/cm² thickness) to mitigate radiation dose, but increases aerodynamic cross-section (drag area +2 m²).
* **[4] Query AI Advice**: Asks Astraea-9000 to analyze current parameters and issue a diagnostic log.
* **[5] Step Sim**: Fetches latest space weather and simulates another orbital period (5400 seconds).
* **[Q] Shutdown**: Disconnects command consoles.

---

## Simulation Physics & Equations

### Atmospheric Drag & Storm Expansion
At altitude $h$ in LEO, the atmospheric density $\rho_0(h)$ is evaluated using an exponential scale height. Geomagnetic storms heat and expand the thermosphere:
$$\rho_{actual} = \rho_0(h) \times \left(1.0 + 0.8 \times G + 0.003 \times \max(0, V_{sw} - 400)\right)$$

The drag acceleration is calculated as:
$$\vec{a}_d = - \frac{1}{2} C_d \frac{A}{m} \rho_{actual} v \vec{v}$$

### Shielding Radiation Attenuation
Solar proton storm flux $I_0$ scales exponentially with the Solar Radiation Scale index $S$:
$$I = I_0 e^{-\alpha \cdot d}$$
where $d$ is shielding thickness, and $\alpha = 0.35$ is the attenuation coefficient.
