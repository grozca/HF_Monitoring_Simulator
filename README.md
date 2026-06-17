# GROZFRAC Frac Monitoring Simulator

A simple professional training simulator for frac monitoring and completion-engineering practice.

The simulator generates a frac stage, applies simplified wellbore/formation physics, and displays the job in a Streamlit dashboard for real-time interpretation practice.

## Current Scope

- Treatment schedule: pad, breakdown, slurry ramp, main proppant, flush
- Curves: rate, surface pressure, PPA, sand rate, pipe friction, perf friction, BHP, net pressure
- Sand transport: surface PPA, bottomhole PPA, sand lag, sand in wellbore, flush arrival
- Formation proxies: fracture width, leakoff, fluid efficiency, acceptance index, screenout risk
- Calibration controls: measured depth to perforations, wellbore capacity, and mixing efficiency
- Monitoring layout: large primary pressure plot plus rate, friction/net, and sand transport panels
- Pressure decomposition: surface pressure, BHP, hydrostatic, pipe friction, perf friction, net pressure
- Well presets: custom, Permian horizontal, high-pressure Delaware, and short training stage
- Field-style stepped PPA schedule option
- Hidden formation response: width, half-length, height, leakoff, fluid efficiency, acceptance, screenout risk, and formation state
- Wellbore hydraulics: MD, TVD, casing ID, capacity, rate BPH, Reynolds/friction proxies, hydrostatic and pipe friction
- Equipment/HHP: pumps, available HHP, required HHP, HHP utilization, pressure margin, rate capacity and equipment status
- Action engine: delayed operational response for reducing PPA, flushing, changing rate, checking pumps, verifying sensors, and evaluating offset communication
- Calibration engine: auto column mapping, time alignment, lag estimation, RMSE/NRMSE, bias, slope error, correlation and pressure-component fit against Excel/CSV reference curves
- Scenarios: normal job, screenout, perforation plugging, pump issue, frac hit, sensor error
- Diagnostics: pressure/rate trend alarms
- Training mode: diagnosis scoring plus action quality, evidence, risk change and response-preview curves

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Engineering Equations

```text
BHP = Surface Pressure + Hydrostatic - Pipe Friction - Perf Friction
BHP Gradient = BHP / TVD
Net Pressure = BHP - Closure Pressure
Sand Rate = Clean Rate * 42 * PPA
Bottomhole PPA(t) = Surface PPA(t - Sand Lag)
Sand Lag = Wellbore Volume / Slurry Rate
Wellbore Volume = Measured Depth * Capacity
Width Proxy = Net Pressure / Rock Stiffness
Formation Acceptance = f(bottomhole PPA, net pressure slope, width, sand loading)
Screenout Risk = f(bottomhole PPA, acceptance, pressure slope, fracture width)
Rate BPH = Rate BPM * 60
HHP Required ~= Surface Pressure * Rate BPM / 40.8
Action Response = 1 - exp(-(t - action_time - delay) / tau)
Residual = Simulation(t) - Reference(t)
NRMSE = RMSE / (P95(reference) - P5(reference))
```

The model is intentionally transparent. It is not a calibrated frac simulator yet; it is a training surface that can later be tied to real Excel treatments and field-derived coefficients.
