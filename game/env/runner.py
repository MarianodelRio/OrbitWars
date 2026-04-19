import json
from kaggle_environments import make
from kaggle_environments.utils import get_player

ORBIT_WARS_RENDERER = """
function renderer({ parent, environment, step, width = 600, height = 600 }) {
  const obs = environment.steps[step][0].observation;
  const planets = obs.planets || [];
  const fleets = obs.fleets || [];
  const BOARD = environment.configuration.boardSize || 100;
  const SUN_R = environment.configuration.sunRadius || 10;

  let canvas = parent.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    parent.appendChild(canvas);
  }
  canvas.width = width;
  canvas.height = height;
  const c = canvas.getContext("2d");
  const scale = width / BOARD;

  // Background
  c.fillStyle = "#0a0a1a";
  c.fillRect(0, 0, width, height);

  // Stars
  const rng = (seed) => { let x = Math.sin(seed) * 10000; return x - Math.floor(x); };
  for (let i = 0; i < 120; i++) {
    c.fillStyle = `rgba(255,255,255,${0.3 + rng(i * 7) * 0.5})`;
    c.beginPath();
    c.arc(rng(i * 3) * width, rng(i * 5) * height, rng(i * 11) * 1.2, 0, Math.PI * 2);
    c.fill();
  }

  // Sun glow
  const cx = BOARD / 2 * scale, cy = BOARD / 2 * scale;
  const glow = c.createRadialGradient(cx, cy, 0, cx, cy, SUN_R * scale * 1.8);
  glow.addColorStop(0, "rgba(255,180,0,0.6)");
  glow.addColorStop(1, "rgba(255,100,0,0)");
  c.fillStyle = glow;
  c.beginPath();
  c.arc(cx, cy, SUN_R * scale * 1.8, 0, Math.PI * 2);
  c.fill();

  // Sun
  c.fillStyle = "#ffaa00";
  c.beginPath();
  c.arc(cx, cy, SUN_R * scale, 0, Math.PI * 2);
  c.fill();

  const playerColors = ["#00ccff", "#ff4444"];
  const neutralColor = "#888888";

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function ownerColor(owner) {
    if (owner === -1) return neutralColor;
    return playerColors[owner] || "#ffffff";
  }

  // Planets
  for (const p of planets) {
    const [id, owner, x, y, radius, ships, production] = p;
    const px = x * scale, py = y * scale, pr = Math.max(radius * scale, 4);
    const color = ownerColor(owner);

    // Planet glow
    const pg = c.createRadialGradient(px, py, 0, px, py, pr * 2);
    pg.addColorStop(0, hexToRgba(color, 0.3));
    pg.addColorStop(1, "rgba(0,0,0,0)");

    // Planet glow arc
    c.fillStyle = pg;
    c.beginPath();
    c.arc(px, py, pr * 2, 0, Math.PI * 2);
    c.fill();

    // Planet body
    c.fillStyle = color;
    c.globalAlpha = owner === -1 ? 0.5 : 0.9;
    c.beginPath();
    c.arc(px, py, pr, 0, Math.PI * 2);
    c.fill();
    c.globalAlpha = 1.0;

    // Border
    c.strokeStyle = owner === -1 ? "#555" : color;
    c.lineWidth = owner === -1 ? 1 : 2;
    c.beginPath();
    c.arc(px, py, pr, 0, Math.PI * 2);
    c.stroke();

    // Ship count label
    c.fillStyle = "#ffffff";
    c.font = `bold ${Math.max(9, pr * 0.7)}px monospace`;
    c.textAlign = "center";
    c.textBaseline = "middle";
    c.fillText(Math.round(ships), px, py);
  }

  // Fleets
  for (const f of fleets) {
    const [id, owner, x, y, angle, from_id, ships] = f;
    const fx = x * scale, fy = y * scale;
    const color = ownerColor(owner);
    c.fillStyle = color;
    c.globalAlpha = 0.85;
    c.beginPath();
    c.arc(fx, fy, 3, 0, Math.PI * 2);
    c.fill();
    c.globalAlpha = 1.0;

    // Ship count next to fleet
    if (ships > 1) {
      c.fillStyle = color;
      c.font = "8px monospace";
      c.textAlign = "left";
      c.textBaseline = "middle";
      c.fillText(Math.round(ships), fx + 5, fy);
    }
  }

  // Step info
  c.fillStyle = "rgba(0,0,0,0.5)";
  c.fillRect(0, 0, 110, 36);
  c.fillStyle = "#ffffff";
  c.font = "12px monospace";
  c.textAlign = "left";
  c.textBaseline = "top";
  c.fillText("Step: " + step, 6, 4);

  // Legend
  const p0ships = planets.filter(p => p[1] === 0).reduce((s, p) => s + p[5], 0)
                + fleets.filter(f => f[1] === 0).reduce((s, f) => s + f[6], 0);
  const p1ships = planets.filter(p => p[1] === 1).reduce((s, p) => s + p[5], 0)
                + fleets.filter(f => f[1] === 1).reduce((s, f) => s + f[6], 0);
  c.fillStyle = playerColors[0];
  c.fillText("P0: " + Math.round(p0ships), 6, 20);
  c.fillStyle = playerColors[1];
  c.fillText("P1: " + Math.round(p1ships), 60, 20);
}
"""


def run_match(bot1, bot2, steps=500, render=False, output_file="game.html"):
    env = make("orbit_wars", configuration={"episodeSteps": steps})
    env.run([bot1, bot2])

    final_obs = env.steps[-1][0]["observation"]
    final_planets = final_obs.get("planets", [])
    final_fleets = env.steps[-1][0]["observation"].get("fleets", [])
    rewards = [
        sum(p[5] for p in final_planets if p[1] == i) + sum(f[6] for f in final_fleets if f[1] == i)
        for i in range(2)
    ]

    if render:
        window_kaggle = {
            "debug": False,
            "playing": False,
            "step": len(env.steps) - 1,
            "controls": True,
            "environment": json.loads(env.render(mode="json")),
            "logs": env.logs,
        }
        html = get_player(window_kaggle, ORBIT_WARS_RENDERER)
        with open(output_file, "w") as f:
            f.write(html)

    if rewards[0] > rewards[1]:
        winner = 0
    elif rewards[1] > rewards[0]:
        winner = 1
    else:
        winner = None

    return {"winner": winner, "rewards": rewards, "steps": len(env.steps)}
