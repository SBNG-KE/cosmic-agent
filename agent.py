import os
import sys
import json
import math
import subprocess
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress

# Load environment variables
load_dotenv()

# Constants
R_EARTH = 6371.0 # km

# Setup console
console = Console()

class SpaceWeatherClient:
    """Client to fetch live space weather data from NOAA SWPC."""
    
    @staticmethod
    def fetch_weather():
        weather = {
            "solar_wind_speed": 400.0,
            "magnetic_field_bt": 5.0,
            "magnetic_field_bz": 0.0,
            "scale_g": 0,
            "scale_s": 0,
            "scale_r": 0,
            "timestamp": "N/A",
            "source": "Default / Cache"
        }
        
        # 1. Fetch NOAA Scales (G, S, R)
        try:
            r = requests.get("https://services.swpc.noaa.gov/products/noaa-scales.json", timeout=4)
            if r.status_code == 200:
                data = r.json()
                # Use index '-1' which contains the most recent observations
                latest = data.get("-1", data.get("0", {}))
                
                weather["timestamp"] = f"{latest.get('DateStamp', '')} {latest.get('TimeStamp', '')}".strip()
                
                # Extract Scale values (defaulting to 0 if None or text 'none')
                for scale_key, target in [("G", "scale_g"), ("S", "scale_s"), ("R", "scale_r")]:
                    scale_data = latest.get(scale_key, {})
                    val_str = scale_data.get("Scale")
                    if val_str and val_str.isdigit():
                        weather[target] = int(val_str)
                weather["source"] = "NOAA Real-time Services"
        except Exception:
            pass
            
        # 2. Fetch Solar Wind Speed
        try:
            r = requests.get("https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json", timeout=4)
            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list) and len(data) > 0:
                    weather["solar_wind_speed"] = float(data[0].get("proton_speed", 400.0))
        except Exception:
            pass
            
        # 3. Fetch Magnetic Field
        try:
            r = requests.get("https://services.swpc.noaa.gov/products/summary/solar-wind-mag-field.json", timeout=4)
            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list) and len(data) > 0:
                    weather["magnetic_field_bt"] = float(data[0].get("bt", 5.0))
                    weather["magnetic_field_bz"] = float(data[0].get("bz_gsm", 0.0))
        except Exception:
            pass
            
        return weather


class AIAgent:
    """Handles LLM communication or local fallback rules for Astraea Agent."""
    
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.hf_key = os.getenv("HF_API_KEY")
        
    def generate_log(self, weather, telemetry, last_action):
        prompt = (
            "You are Astraea-9000, an autonomous AI companion and ship computer on a research satellite orbiting Earth.\n"
            "Analyze current space weather conditions and the spacecraft's physics telemetry to write a short ship log entry.\n\n"
            f"--- Space Weather ---\n"
            f"- Solar Wind Speed: {weather['solar_wind_speed']} km/s (Normal: 300-450 km/s)\n"
            f"- Interplanetary Magnetic Field Bt: {weather['magnetic_field_bt']} nT\n"
            f"- NOAA Scales: Geomagnetic Storm (G{weather['scale_g']}), Solar Radiation Storm (S{weather['scale_s']}), Radio Blackout (R{weather['scale_r']})\n\n"
            f"--- Spacecraft Telemetry ---\n"
            f"- Altitude: {telemetry['altitude_km']:.2f} km\n"
            f"- Velocity: {telemetry['velocity_kms']:.3f} km/s\n"
            f"- Orbit Status: {telemetry['orbit_status'].upper()}\n"
            f"- Altitude decay during last orbit: {telemetry['altitude_decay_m']:.1f} meters\n"
            f"- Fuel: {telemetry['fuel_kg']:.1f} kg\n"
            f"- Shielding Thickness: {telemetry['shielding']:.2f} g/cm2\n"
            f"- Present Dose Rate: {telemetry['dose_rate_usv_hr']:.3f} uSv/hr\n"
            f"- Accumulated Mission Dose: {telemetry['accumulated_dose_usv']:.3f} uSv\n\n"
            f"--- Context ---\n"
            f"- Last Command Initiated: {last_action}\n\n"
            "Instructions:\n"
            "Write a log entry from your perspective (Astraea-9000). Keep it strictly within 3-4 sentences. "
            "Sound intelligent, slightly robotic, but protective. Address the telemetry directly: note any high radiation "
            "(S scale), increased orbital decay from thermospheric expansion (G scale / high solar winds), or fuel constraints. "
            "Suggest the next logical action (e.g., boost orbit, deploy shielding, or maintain course)."
        )
        
        # 1. Try Gemini
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                pass
                
        # 2. Try OpenAI
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                pass
                
        # 3. Try Hugging Face Serverless
        if self.hf_key:
            try:
                headers = {"Authorization": f"Bearer {self.hf_key}"}
                API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
                payload = {
                    "inputs": f"<s>[INST] {prompt} [/INST]",
                    "parameters": {"max_new_tokens": 150, "temperature": 0.7}
                }
                r = requests.post(API_URL, headers=headers, json=payload, timeout=6)
                if r.status_code == 200:
                    text = r.json()[0]['generated_text']
                    if "[/INST]" in text:
                        text = text.split("[/INST]")[-1].strip()
                    return text
            except Exception:
                pass
                
        # 4. Fallback Rule-Based Engine
        return self._generate_rule_based_log(weather, telemetry, last_action)

    def _generate_rule_based_log(self, weather, telemetry, last_action):
        logs = []
        logs.append(f"[Astraea-9000 Log - Stardate local-time]")
        
        # Action comment
        if last_action == "boost_low":
            logs.append("Executed low prograde thruster burn. Orbit raised successfully, expending 10kg monopropellant.")
        elif last_action == "boost_high":
            logs.append("Executed major prograde thruster burn. Altitude restored significantly, expending 30kg monopropellant.")
        elif last_action == "shield_deploy":
            logs.append("Radiation deflectors deployed. Structural shielding increased, though aerodynamic drag cross-section is expanded.")
        
        # Altitude / Drag warnings
        decay = telemetry['altitude_decay_m']
        alt = telemetry['altitude_km']
        g = weather['scale_g']
        wind = weather['solar_wind_speed']
        
        if telemetry['orbit_status'] == "reentered":
            logs.append("CRITICAL ERROR: Orbital decay exceeded bounds. Spacecraft has entered the dense atmosphere and disintegrated.")
        elif alt < 250.0:
            logs.append(f"CRITICAL WARNING: Altitude critical at {alt:.1f}km. Atmospheric drag is severe. Immediate orbital boost is required.")
        elif g >= 2 or wind > 600.0:
            logs.append(f"Space weather activity (G{g} storm / wind {wind:.0f} km/s) is heating the thermosphere. Orbital decay has accelerated to {decay:.1f}m per orbit.")
        else:
            logs.append(f"Orbital parameters stable. Decay rate is nominal at {decay:.1f}m per orbit under standard solar wind conditions.")
            
        # Radiation warning
        s = weather['scale_s']
        dose_rate = telemetry['dose_rate_usv_hr']
        if s >= 2:
            logs.append(f"CAUTION: Solar radiation storm level S{s} detected. Onboard dose rate elevated to {dose_rate:.2f} uSv/hr. Shielding recommended.")
        elif dose_rate > 10.0:
            logs.append(f"Dose rate elevated at {dose_rate:.2f} uSv/hr. Avionic shielding integrity is holding.")
            
        # Fuel warning
        fuel = telemetry['fuel_kg']
        if fuel < 20.0 and telemetry['orbit_status'] != "reentered":
            logs.append("WARNING: Monopropellant reserve levels below 20%. Recommend conservation of fuel resources.")
            
        return " ".join(logs)


def draw_ascii_orbit(x, y, status):
    """Generates an ASCII diagram showing Earth and the satellite's position."""
    width = 33
    height = 15
    grid = [[" " for _ in range(width)] for _ in range(height)]
    cx, cy = width // 2, height // 2
    
    # Draw Earth core
    for r in range(height):
        for c in range(width):
            dx = (c - cx) * 0.55 # Scale for characters aspect ratio
            dy = (r - cy)
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 2.5:
                grid[r][c] = "🌍" if (c % 2 == 0) else ""
            elif dist < 3.2:
                grid[r][c] = "."
                
    # Draw Satellite
    if status != "reentered" and (x != 0.0 or y != 0.0):
        # Scale actual coordinate to fits grid
        # RE ~ 6371km. Satellite at ~ 6771km.
        # We want Earth radius to map to dist ~ 3.0, satellite to ~ 4.2
        scale = 3.0 / 6371000.0
        sat_x = cx + int(round(x * scale * 1.8)) # aspect ratio correction
        sat_y = cy - int(round(y * scale))
        
        # Clamp within grid
        sat_x = max(0, min(width - 1, sat_x))
        sat_y = max(0, min(height - 1, sat_y))
        grid[sat_y][sat_x] = "🛰️"
        
    lines = []
    for r in range(height):
        line = "".join([ch if ch != "" else " " for ch in grid[r]])
        lines.append(line)
    return "\n".join(lines)


def run_rust_simulation(state, weather, maneuver):
    """Executes the Rust cosmic_physics binary with current state and weather parameters."""
    binary_path = os.path.join("target", "release", "cosmic_physics.exe")
    if not os.path.exists(binary_path):
        # Fallback to debug if release doesn't exist
        binary_path = os.path.join("target", "debug", "cosmic_physics.exe")
        if not os.path.exists(binary_path):
            console.print("[bold red]Error: Rust simulation engine executable not found. Please compile the project first using 'cargo build --release'[/bold red]")
            sys.exit(1)
            
    input_data = {
        "x": state.get("x", 0.0),
        "y": state.get("y", 0.0),
        "vx": state.get("vx", 0.0),
        "vy": state.get("vy", 0.0),
        "initial_altitude_km": state.get("altitude_km", 400.0),
        "mass": state.get("mass", 500.0),
        "drag_area": state.get("drag_area", 4.0),
        "shielding": state.get("shielding", 2.0),
        "fuel": state.get("fuel", 100.0),
        "solar_wind_speed": weather["solar_wind_speed"],
        "magnetic_field_bt": weather["magnetic_field_bt"],
        "scale_g": weather["scale_g"],
        "scale_s": weather["scale_s"],
        "maneuver": maneuver,
        "sim_duration_sec": 5400 # 1 orbit step
    }
    
    try:
        proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(input=json.dumps(input_data))
        if proc.returncode != 0:
            console.print(f"[bold red]Rust Engine Error: {stderr}[/bold red]")
            sys.exit(1)
        return json.loads(stdout)
    except Exception as e:
        console.print(f"[bold red]Failed to execute Rust simulation engine: {e}[/bold red]")
        sys.exit(1)


def main():
    console.clear()
    console.print("[bold cyan]ASTRAEA COMMAND DASHBOARD INITIALIZING...[/bold cyan]")
    
    # Initialize state
    state = {
        "x": 0.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "altitude_km": 400.0,
        "mass": 500.0,
        "drag_area": 4.0,
        "shielding": 2.0,
        "fuel": 100.0,
        "accumulated_dose_usv": 0.0,
        "dose_rate_usv_hr": 0.0,
        "orbit_status": "stable",
        "altitude_decay_m": 0.0
    }
    
    # Initialize weather
    weather = SpaceWeatherClient.fetch_weather()
    agent = AIAgent()
    
    # Run initial simulation to populate orbital variables (x, y, vx, vy)
    state = run_rust_simulation(state, weather, "none")
    
    last_action = "Initialization"
    log_entry = agent.generate_log(weather, state, last_action)
    
    while True:
        # Build UI layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right", ratio=2)
        )
        
        layout["right"].split_column(
            Layout(name="top_right"),
            Layout(name="bottom_right", size=8)
        )
        
        # Render Header
        status_text = "[bold green]ALL SYSTEMS NOMINAL[/bold green]"
        if state["orbit_status"] == "reentered":
            status_text = "[bold blink red]SPACECRAFT DE-ORBITED / DESTROYED[/bold blink red]"
        elif state["orbit_status"] == "decaying":
            status_text = "[bold yellow]ACCELERATED ORBIT DECAY[/bold yellow]"
        elif state["altitude_km"] < 250.0:
            status_text = "[bold blink red]CRITICAL ALTITUDE WARNING[/bold blink red]"
            
        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left", ratio=1)
        header_table.add_column(justify="right", ratio=1)
        header_table.add_row(
            Text.from_markup("[bold white]🛰️ ASTRAEA COMMAND CONSOLE v1.0.0[/bold white]"),
            Text.from_markup(f"SYSTEM STATUS: {status_text}")
        )
        layout["header"].update(Panel(header_table, style="cyan"))
        
        # Render Left Panel: Space Weather
        weather_table = Table.grid(padding=1)
        weather_table.add_column(style="bold white", width=18)
        weather_table.add_column(style="cyan")
        
        speed_col = "green" if weather["solar_wind_speed"] < 500 else ("yellow" if weather["solar_wind_speed"] < 700 else "red")
        bt_col = "green" if weather["magnetic_field_bt"] < 10 else ("yellow" if weather["magnetic_field_bt"] < 20 else "red")
        
        weather_table.add_row("Solar Wind Speed:", f"[{speed_col}]{weather['solar_wind_speed']:.1f} km/s[/{speed_col}]")
        weather_table.add_row("IMF Bt Magnitude:", f"[{bt_col}]{weather['magnetic_field_bt']:.1f} nT[/{bt_col}]")
        weather_table.add_row("IMF Bz (GSM):", f"{weather['magnetic_field_bz']:.1f} nT")
        weather_table.add_row("NOAA Scale G (Geo):", f"G{weather['scale_g']} (0-5)")
        weather_table.add_row("NOAA Scale S (Rad):", f"S{weather['scale_s']} (0-5)")
        weather_table.add_row("NOAA Scale R (Rad):", f"R{weather['scale_r']} (0-5)")
        weather_table.add_row("Data Source:", f"[dim]{weather['source']}[/dim]")
        weather_table.add_row("Last Update:", f"[dim]{weather['timestamp']}[/dim]")
        
        layout["left"].update(Panel(weather_table, title="[bold yellow]📡 Real-time Space Weather (NOAA)[/bold yellow]", style="yellow"))
        
        # Render Top Right: Telemetry + Orbit Map
        telemetry_table = Table.grid(padding=1)
        telemetry_table.add_column(style="bold white", width=22)
        telemetry_table.add_column(style="cyan", width=12)
        
        alt_col = "green" if state["altitude_km"] > 350 else ("yellow" if state["altitude_km"] > 250 else "red")
        fuel_col = "green" if state["fuel_kg"] > 50 else ("yellow" if state["fuel_kg"] > 15 else "red")
        dose_col = "green" if state["dose_rate_usv_hr"] < 5.0 else ("yellow" if state["dose_rate_usv_hr"] < 30.0 else "red")
        
        telemetry_table.add_row("Altitude:", f"[{alt_col}]{state['altitude_km']:.2f} km[/{alt_col}]")
        telemetry_table.add_row("Velocity:", f"{state['velocity_kms']:.3f} km/s")
        telemetry_table.add_row("Decay Rate / Orbit:", f"{state['altitude_decay_m']:.1f} meters")
        telemetry_table.add_row("Monopropellant Fuel:", f"[{fuel_col}]{state['fuel_kg']:.1f} kg[/{fuel_col}]")
        telemetry_table.add_row("Shielding Thickness:", f"{state['shielding']:.2f} g/cm²")
        telemetry_table.add_row("Dose Rate:", f"[{dose_col}]{state['dose_rate_usv_hr']:.3f} uSv/hr[/{dose_col}]")
        telemetry_table.add_row("Accumulated Dose:", f"{state['accumulated_dose_usv']:.3f} uSv")
        telemetry_table.add_row("Orbit Status:", f"[bold]{state['orbit_status'].upper()}[/bold]")
        
        # Draw Orbit Diagram
        orbit_ascii = draw_ascii_orbit(state["x"], state["y"], state["orbit_status"])
        
        combined_row_table = Table.grid(expand=True)
        combined_row_table.add_column(ratio=1)
        combined_row_table.add_column(ratio=1)
        combined_row_table.add_row(telemetry_table, orbit_ascii)
        
        layout["top_right"].update(Panel(combined_row_table, title="[bold green]📊 Spacecraft Telemetry & Orbit Visualization[/bold green]", style="green"))
        
        # Render Bottom Right: Onboard AI Companion
        layout["bottom_right"].update(Panel(
            Text(log_entry, style="italic green"),
            title=f"[bold green]🤖 Onboard AI Computer: Astraea-9000[/bold green]",
            style="green"
        ))
        
        # Render Footer (Commands menu)
        footer_text = (
            "[bold white]COMMAND MENU:[/bold white] "
            "[bold green][1][/bold green] Boost Orbit (Low)  "
            "[bold green][2][/bold green] Boost Orbit (High)  "
            "[bold green][3][/bold green] Deploy Radiation Shielding  "
            "[bold green][4][/bold green] Query AI Advice  "
            "[bold green][5][/bold green] Fetch Weather & Step Sim (1 Orbit)  "
            "[bold red][Q][/bold red] Shutdown"
        )
        layout["footer"].update(Panel(Text.from_markup(footer_text), style="cyan"))
        
        console.clear()
        console.print(layout)
        
        if state["orbit_status"] == "reentered":
            console.print("[bold red]Spacecraft has re-entered. Simulation ended.[/bold red]")
            break
            
        # Get user command
        try:
            cmd = input("\nEnter command option: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            cmd = 'q'
            
        if cmd == 'q':
            console.print("[bold cyan]Shutting down Astraea systems... Goodbye.[/bold cyan]")
            break
            
        maneuver = "none"
        last_action = "Simulated 1 Orbit (Weather Refreshed)"
        
        if cmd == '1':
            maneuver = "boost_low"
            last_action = "Initiated Low Prograde Burn"
        elif cmd == '2':
            maneuver = "boost_high"
            last_action = "Initiated High Prograde Burn"
        elif cmd == '3':
            maneuver = "shield_deploy"
            last_action = "Deployed Radiation Deflector Shielding"
        elif cmd == '4':
            console.print("[yellow]Contacting Astraea-9000 for advice...[/yellow]")
            log_entry = agent.generate_log(weather, state, "AI Consultation Request")
            continue
        elif cmd == '5':
            # Just refreshing weather and stepping
            pass
        else:
            console.print("[yellow]Invalid option. Press enter to continue...[/yellow]")
            input()
            continue
            
        # Execute step
        console.print("[yellow]Executing simulation step (5400 seconds orbital path)...[/yellow]")
        weather = SpaceWeatherClient.fetch_weather() # Get latest NOAA weather
        state = run_rust_simulation(state, weather, maneuver)
        log_entry = agent.generate_log(weather, state, last_action)

if __name__ == "__main__":
    main()
