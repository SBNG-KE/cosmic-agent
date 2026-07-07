use serde::{Deserialize, Serialize};
use std::io::{self, Read};

const RE: f64 = 6371000.0; // Earth radius in meters
const MU: f64 = 3.986004418e14; // Earth gravitational parameter (m^3/s^2)

#[derive(Deserialize, Serialize, Debug, Clone)]
struct SimInput {
    #[serde(default)]
    x: f64,
    #[serde(default)]
    y: f64,
    #[serde(default)]
    vx: f64,
    #[serde(default)]
    vy: f64,
    #[serde(default = "default_altitude")]
    initial_altitude_km: f64,
    #[serde(default = "default_mass")]
    mass: f64,
    #[serde(default = "default_drag_area")]
    drag_area: f64,
    #[serde(default = "default_shielding")]
    shielding: f64,
    #[serde(default = "default_fuel")]
    fuel: f64,
    #[serde(default = "default_solar_wind")]
    solar_wind_speed: f64,
    #[serde(default = "default_bt")]
    magnetic_field_bt: f64,
    #[serde(default)]
    scale_g: u32,
    #[serde(default)]
    scale_s: u32,
    #[serde(default = "default_maneuver")]
    maneuver: String,
    #[serde(default = "default_duration")]
    sim_duration_sec: u32,
}

fn default_altitude() -> f64 { 400.0 }
fn default_mass() -> f64 { 500.0 }
fn default_drag_area() -> f64 { 4.0 }
fn default_shielding() -> f64 { 2.0 }
fn default_fuel() -> f64 { 100.0 }
fn default_solar_wind() -> f64 { 400.0 }
fn default_bt() -> f64 { 5.0 }
fn default_maneuver() -> String { "none".to_string() }
fn default_duration() -> u32 { 5400 } // 1.5 hours (approx 1 orbit)

#[derive(Serialize, Debug)]
struct SimOutput {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    altitude_km: f64,
    velocity_kms: f64,
    fuel_kg: f64,
    shielding: f64,
    accumulated_dose_usv: f64,
    dose_rate_usv_hr: f64,
    orbit_status: String,
    altitude_decay_m: f64,
    orbit_path: Vec<(f64, f64)>,
}

fn get_standard_density(altitude_meters: f64) -> f64 {
    let h = altitude_meters / 1000.0;
    if h < 100.0 {
        return 1.225;
    }
    // Piecewise exponential atmospheric density model for LEO thermosphere
    if h < 150.0 {
        2.0e-9 * (-(h - 100.0) / 10.0).exp()
    } else if h < 250.0 {
        3.0e-10 * (-(h - 150.0) / 25.0).exp()
    } else if h < 400.0 {
        2.4e-11 * (-(h - 250.0) / 50.0).exp()
    } else if h < 600.0 {
        1.5e-12 * (-(h - 400.0) / 80.0).exp()
    } else if h < 800.0 {
        1.2e-13 * (-(h - 600.0) / 120.0).exp()
    } else {
        1.0e-14 * (-(h - 800.0) / 200.0).exp()
    }
}

fn main() {
    // Read input JSON from stdin
    let mut buffer = String::new();
    if io::stdin().read_to_string(&mut buffer).is_err() {
        eprintln!("Failed to read from stdin");
        std::process::exit(1);
    }

    let mut input: SimInput = match serde_json::from_str(&buffer) {
        Ok(val) => val,
        Err(e) => {
            eprintln!("Failed to parse JSON: {}", e);
            std::process::exit(1);
        }
    };

    // Initialize orbit if position/velocity are not set
    if input.x == 0.0 && input.y == 0.0 {
        let r = RE + (input.initial_altitude_km * 1000.0);
        input.x = 0.0;
        input.y = r;
        // Circular orbit velocity: v = sqrt(MU / r)
        let v = (MU / r).sqrt();
        input.vx = v;
        input.vy = 0.0;
    }

    // Apply maneuvers at start of step
    let mut drag_area_modifier = 0.0;
    match input.maneuver.as_str() {
        "boost_low" => {
            if input.fuel >= 10.0 {
                input.fuel -= 10.0;
                // Add delta-v along velocity vector direction
                let speed = (input.vx * input.vx + input.vy * input.vy).sqrt();
                if speed > 0.0 {
                    input.vx += (input.vx / speed) * 25.0;
                    input.vy += (input.vy / speed) * 25.0;
                }
            }
        }
        "boost_high" => {
            if input.fuel >= 30.0 {
                input.fuel -= 30.0;
                let speed = (input.vx * input.vx + input.vy * input.vy).sqrt();
                if speed > 0.0 {
                    input.vx += (input.vx / speed) * 75.0;
                    input.vy += (input.vy / speed) * 75.0;
                }
            }
        }
        "shield_deploy" => {
            // Increases shielding thickness but also increases drag area
            input.shielding += 1.5;
            drag_area_modifier = 2.0;
        }
        _ => {}
    }

    // Density factor scaling with Space Weather:
    // Scale G is geomagnetic storm scale (0 to 5)
    // Solar wind speed (typical 300 - 400, solar storms can exceed 800)
    let wind_excess = (input.solar_wind_speed - 400.0).max(0.0);
    let density_multiplier = 1.0 + (input.scale_g as f64 * 0.8) + (wind_excess * 0.003);

    // Dose rate calculation based on Radiation scale S
    // S is solar radiation scale (0 to 5)
    // Shielding reduces radiation exponentially
    let base_dose_rate = 0.05 + (input.scale_s as f64).powf(1.8) * 15.0; // uSv/hr
    let dose_rate_usv_hr = base_dose_rate * (-0.35 * input.shielding).exp();

    let dt = 5.0; // 5 second step size
    let steps = input.sim_duration_sec / (dt as u32);
    let mut x = input.x;
    let mut y = input.y;
    let mut vx = input.vx;
    let mut vy = input.vy;
    let mut accumulated_dose_usv = 0.0;
    let mut orbit_path = Vec::new();
    let mut status = "stable".to_string();

    let initial_r = (x*x + y*y).sqrt();
    let initial_alt = initial_r - RE;

    let path_sample_interval = (steps / 20).max(1);

    for step in 0..steps {
        let r = (x * x + y * y).sqrt();
        let altitude = r - RE;

        if altitude < 100000.0 {
            status = "reentered".to_string();
            x = 0.0;
            y = 0.0;
            vx = 0.0;
            vy = 0.0;
            break;
        }

        // Atmosphere Density with weather scale modifier
        let raw_density = get_standard_density(altitude);
        let actual_density = raw_density * density_multiplier;

        // Velocity magnitude
        let v = (vx * vx + vy * vy).sqrt();

        // Gravitational acceleration (m/s^2)
        let ax_g = -MU * x / (r * r * r);
        let ay_g = -MU * y / (r * r * r);

        // Drag acceleration (m/s^2)
        let effective_drag_area = input.drag_area + drag_area_modifier;
        let ad_mag = 0.5 * 2.2 * effective_drag_area * actual_density * v * v / input.mass;
        let ax_d = if v > 0.0 { -ad_mag * (vx / v) } else { 0.0 };
        let ay_d = if v > 0.0 { -ad_mag * (vy / v) } else { 0.0 };

        // Total acceleration
        let ax = ax_g + ax_d;
        let ay = ay_g + ay_d;

        // Integration (Euler-Cromer)
        vx += ax * dt;
        vy += ay * dt;
        x += vx * dt;
        y += vy * dt;

        // Radiation dose accumulation over this time step
        accumulated_dose_usv += (dose_rate_usv_hr * dt) / 3600.0;

        // Record path coordinates periodically (scaled for TUI display context)
        if step % path_sample_interval == 0 {
            orbit_path.push((x, y));
        }
    }

    let final_r = (x*x + y*y).sqrt();
    let final_alt = if status == "reentered" { 0.0 } else { final_r - RE };
    let final_speed = if status == "reentered" { 0.0 } else { (vx*vx + vy*vy).sqrt() };
    let altitude_decay_m = if status == "reentered" { initial_alt } else { initial_alt - final_alt };

    if status == "stable" && altitude_decay_m > 500.0 {
        status = "decaying".to_string();
    }

    let output = SimOutput {
        x,
        y,
        vx,
        vy,
        altitude_km: final_alt / 1000.0,
        velocity_kms: final_speed / 1000.0,
        fuel_kg: input.fuel,
        shielding: input.shielding,
        accumulated_dose_usv,
        dose_rate_usv_hr,
        orbit_status: status,
        altitude_decay_m,
        orbit_path,
    };

    let output_json = serde_json::to_string_pretty(&output).unwrap();
    println!("{}", output_json);
}
