from __future__ import annotations

import argparse
import math

import numpy as np
import pyglet
from pyglet import shapes, text
from pyglet.math import Mat4
from pyglet.window import key, mouse

from .config import Config
from .experiments import list_experiment_presets, run_parallel_experiment, run_preset_comparison
from .simulation import Agent, Simulation


# Color palette - brightened for visibility
WORLD_BG = (20, 25, 35)
PANEL_BG = (18, 22, 28)
GRID = (60, 68, 82)
FOOD = (100, 235, 145)
FOOD_ALT = (70, 195, 120)
AGENT = (255, 185, 110)
AGENT_BORDER = (255, 230, 195)
SELECTED = (120, 200, 255)
SELECTED_GLOW = (80, 170, 255)
TEXT_COLOR = (230, 235, 245)
MUTED = (145, 155, 170)
ACCENT = (115, 230, 200)
PAUSED_OVERLAY = (0, 0, 0)

LINEAGE_PALETTE = [
    (230, 92, 92),
    (92, 146, 230),
    (232, 176, 72),
    (118, 214, 160),
    (210, 120, 220),
    (220, 150, 92),
    (120, 198, 240),
    (186, 96, 140),
    (174, 202, 94),
    (104, 168, 218),
]


class PrimordiaWindow(pyglet.window.Window):
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

        # Make the total window taller so the sidebar (with Best of Run + Brain State + Memory + Controls)
        # has enough vertical space and doesn't get cut off.
        window_height = 980   # increased from 720 to give sidebar breathing room

        super().__init__(
            width=self.cfg.world_width + 320,
            height=window_height,
            caption="Primordia — Early Life Simulator",
            resizable=True,          # allow user to resize if they want even more space
            vsync=True,
        )

        # Simulation
        self.simulation = Simulation(self.cfg)
        self.selected_entity: tuple[str, object] | None = None
        self.show_help = True
        self.show_run_data = True

        # Camera / zoom state for world rendering
        self.zoom = 1.2
        self.camera_x = max(0.0, (self.cfg.world_width - (self.cfg.world_width / self.zoom)) * 0.5)
        self.camera_y = max(0.0, (self.cfg.world_height - (self.cfg.world_height / self.zoom)) * 0.5)
        self.min_zoom = 0.25
        self.max_zoom = 8.0
        self._camera_dragging = False
        self._camera_drag_start: tuple[float, float] | None = None
        self._camera_drag_origin: tuple[float, float] | None = None
        self._camera_drag_threshold = 4.0

        # Cached viewport rect for the fixed-aspect world area
        self._world_viewport: tuple[int, int, int, int] | None = None  # (x, y, width, height)

        # Clock: step simulation at fixed rate (independent of rendering)
        pyglet.clock.schedule_interval(self._simulation_step, 1.0 / 60.0)

        # Key state
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)

    def _simulation_step(self, dt: float):
        if self.simulation.paused:
            return

        # Respect time scaling (smooth +/- control)
        steps = max(1, int(round(self.simulation.time_scale)))
        for _ in range(steps):
            self.simulation.step()

            # Safety: stop if population dies during fast forward
            if len(self.simulation.agents) == 0:
                break

            # Clear dead selection
            if self.selected_entity is not None and self._get_selected_entity() is None:
                self.selected_entity = None

    def on_draw(self):
        # Enable blending — critical for pyglet 2.x shapes to render correctly
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

        # Clear with background color
        pyglet.gl.glClearColor(WORLD_BG[0]/255, WORLD_BG[1]/255, WORLD_BG[2]/255, 1)
        self.clear()

        self._draw_world()
        self._draw_sidebar()

        # Subtle, non-blocking pause indicator (top-left of world viewport)
        if self.simulation.paused:
            vp_x, vp_y, vp_w, vp_h = self._get_world_viewport()
            banner = shapes.Rectangle(vp_x + 8, vp_y + vp_h - 38, 140, 30, color=(30, 30, 35))
            banner.opacity = 200
            banner.draw()
            text.Label(
                "⏸  PAUSED",
                font_name="Segoe UI",
                font_size=16,
                color=(255, 180, 120, 255),
                x=vp_x + 18,
                y=vp_y + vp_h - 28,
            ).draw()

    def _draw_world(self):
        w = self.cfg.world_width
        h = self.cfg.world_height
        debug_overlays = self.cfg.show_debug_overlays and len(self.simulation.agents) <= self.cfg.debug_overlay_agent_threshold

        # Make sure blending is on
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

        # Compute viewport and apply correct aspect-preserving camera transform
        vp_x, vp_y, vp_w, vp_h = self._get_world_viewport()

        original_view = self.view

        scale = vp_w / w
        eff_scale = scale * self.zoom

        self.view = (
            Mat4().translate((vp_x, vp_y, 0)) @
            Mat4().scale((eff_scale, eff_scale, 1)) @
            Mat4().translate((-self.camera_x, -self.camera_y, 0))
        )

        # === World drawing (all in the transformed space) ===
        # World background
        shapes.Rectangle(0, 0, w, h, color=WORLD_BG).draw()

        # Grid
        step = 80
        for x in range(0, w + 1, step):
            shapes.Line(x, 0, x, h, 1, color=GRID).draw()
        for y in range(0, h + 1, step):
            shapes.Line(0, y, w, y, 1, color=GRID).draw()

        # Toxicity heatmap overlay
        for zone in self.simulation.toxic_zones:
            base_color = (255, 90, 90)
            for band in range(7, 0, -1):
                band_radius = zone.radius * (band / 7.0)
                alpha = int(18 + zone.strength * 18 * band)
                heat = shapes.Circle(zone.x, zone.y, band_radius, color=base_color)
                heat.opacity = max(10, min(90, alpha))
                heat.draw()

            core = shapes.Circle(zone.x, zone.y, max(8.0, zone.radius * 0.22), color=(255, 120, 120))
            core.opacity = 150
            core.draw()

        # Terrain blocks / habitat rocks
        for block in self.simulation.blocks:
            if block.kind == "hard":
                color = (72, 72, 78)
            else:
                color = (172, 176, 184)

            rock = shapes.Circle(block.x, block.y, block.radius, color=color)
            rock.opacity = 235 if block.kind == "hard" else 210
            rock.draw()

            if block.habitat_bonus > 0:
                halo = shapes.Circle(block.x, block.y, block.radius + 14, color=(150, 190, 230))
                halo.opacity = int(40 + 80 * block.habitat_bonus)
                halo.draw()

        # Food
        for food in self.simulation.foods:
            if food.kind == "meat":
                color = (205, 145, 95)
            else:
                color = FOOD if int(food.x + food.y) % 2 == 0 else FOOD_ALT

            shapes.Circle(
                food.x, food.y,
                food.radius + 0.8,
                color=color
            ).draw()

        # Pheromone / signal overlay
        if self.simulation.signals:
            visible_signals = self.simulation.signals[-80:]
            for signal in visible_signals:
                if signal.kind == "danger":
                    color = (255, 92, 92)
                elif signal.kind == "mate":
                    color = (120, 180, 255)
                else:
                    color = (110, 235, 150)

                intensity = max(0.08, min(1.0, signal.strength))
                radius = 16 + intensity * 62

                halo = shapes.Circle(signal.x, signal.y, radius, color=color)
                halo.opacity = int(55 * intensity)
                halo.draw()

                ring = shapes.Circle(signal.x, signal.y, radius * 0.55, color=color)
                ring.opacity = int(90 * intensity)
                ring.draw()

                core = shapes.Circle(signal.x, signal.y, max(3.0, radius * 0.16), color=(245, 248, 255))
                core.opacity = int(130 * intensity)
                core.draw()

        # Agents
        selected = self._get_selected_entity()

        for agent in self.simulation.agents:
            r = self.cfg.agent_radius * (0.82 + agent.genome.genes.size * 0.88)
            r = max(4.2, r)
            energy_ratio = min(1.0, max(0.0, agent.energy / self.cfg.agent_max_energy))
            health_ratio = min(1.0, max(0.0, agent.health / self.cfg.agent_max_health))

            # Strong, distinct colors by lineage.
            base_r, base_g, base_b = self._lineage_color(agent.lineage)

            success = min(1.0, agent.offspring_count / 10.0)

            size_tone = min(1.0, agent.genome.genes.size / 2.4)
            r_col = int(base_r + (255 - base_r) * success * 0.18 - 32 * (1 - energy_ratio) + 10 * size_tone)
            g_col = int(base_g + (255 - base_g) * success * 0.18 + 18 * energy_ratio - 12 * size_tone)
            b_col = int(base_b + (255 - base_b) * success * 0.18 - 10 * (1 - health_ratio) + 18 * size_tone)

            body_color = (
                max(40, min(255, r_col)),
                max(40, min(255, g_col)),
                max(40, min(255, b_col))
            )

            shapes.Circle(agent.x, agent.y, r, color=body_color).draw()

            border = shapes.Circle(agent.x, agent.y, r, color=AGENT_BORDER)
            border.opacity = 170
            border.draw()

            if self.cfg.show_velocity_lines:
                speed = math.hypot(agent.vx, agent.vy)
                if speed > 0.12:
                    line_len = 8.0 + min(11.0, speed * 3.8)
                    vx = agent.x + agent.vx * line_len
                    vy = agent.y + agent.vy * line_len
                    thickness = 2.6 if speed > 0.7 else 2.1
                    shapes.Line(agent.x, agent.y, vx, vy, thickness, color=(248, 250, 255)).draw()

            # Selection highlight
            if selected and selected[0] == "agent" and selected[1] is agent:
                glow1 = shapes.Circle(agent.x, agent.y, r + 10, color=SELECTED_GLOW)
                glow1.opacity = 85
                glow1.draw()

                glow2 = shapes.Circle(agent.x, agent.y, r + 5.5, color=SELECTED)
                glow2.opacity = 210
                glow2.draw()

            if debug_overlays:
                # Memory glow
                if len(agent.memory) > 0:
                    mem_activity = float(np.mean(np.abs(agent.memory)))
                    if mem_activity > 0.08:
                        glow_radius = r + 13 + min(6, mem_activity * 8)
                        intensity = min(0.7, mem_activity * 1.3)
                        mem_glow = shapes.Circle(agent.x, agent.y, glow_radius, color=(80, 160, 255))
                        mem_glow.opacity = int(90 * intensity)
                        mem_glow.draw()

                # Elite ring
                if agent.elite_remaining > 0:
                    elite_alpha = min(180, int(120 * (agent.elite_remaining / 400.0)))
                    elite_ring = shapes.Circle(agent.x, agent.y, r + 15, color=(255, 215, 80))
                    elite_ring.opacity = elite_alpha
                    elite_ring.draw()

        if selected and selected[0] == "food":
            food = selected[1]
            glow = shapes.Circle(food.x, food.y, food.radius + 10, color=SELECTED_GLOW)
            glow.opacity = 110
            glow.draw()

        if selected and selected[0] == "block":
            block = selected[1]
            glow = shapes.Circle(block.x, block.y, block.radius + 12, color=SELECTED_GLOW)
            glow.opacity = 110
            glow.draw()

        if selected and selected[0] == "zone":
            zone = selected[1]
            glow = shapes.Circle(zone.x, zone.y, zone.radius + 10, color=SELECTED_GLOW)
            glow.opacity = 90
            glow.draw()

        # Sensor rays for selected agent
        if selected and selected[0] == "agent" and debug_overlays:
            rays = self.simulation.get_sensor_rays(selected[1])
            for ray in rays:
                if ray.get('inactive'):
                    color = (70, 80, 95)
                    width = 1.0
                elif ray['hit_food']:
                    color = (100, 240, 130)
                    width = 2.2
                else:
                    color = (160, 170, 185)
                    width = 1.3

                shapes.Line(
                    ray['start'][0], ray['start'][1],
                    ray['end'][0], ray['end'][1],
                    width, color=color
                ).draw()

        # Predation flashes
        if self.simulation.predation_flashes:
            remaining = []
            for x, y, life in self.simulation.predation_flashes:
                if life > 0:
                    alpha = min(1.0, life / 14.0)
                    radius = 16 + (14 - life) * 2.0
                    flash = shapes.Circle(x, y, radius, color=(230, 50, 50))
                    flash.opacity = int(150 * alpha)
                    flash.draw()
                    remaining.append((x, y, life - 1))
            self.simulation.predation_flashes = remaining

        # Restore screen-space view for sidebar
        self.view = original_view

    def _draw_sidebar(self):
        x0 = self.cfg.world_width
        w = 320
        h = self.height          # use actual window height (now taller)
        debug_overlays = self.cfg.show_debug_overlays and len(self.simulation.agents) <= self.cfg.debug_overlay_agent_threshold

        # Sidebar background - full height of the window
        sidebar = shapes.Rectangle(x0, 0, w, h, color=PANEL_BG)
        sidebar.draw()

        # Divider
        shapes.Line(x0, 0, x0, h, 2, color=GRID).draw()

        # Text
        y = h - 24
        label_x = x0 + 18

        # Title
        text.Label(
            "PRIMORDIA", font_name="Segoe UI", font_size=18,
            color=(*ACCENT, 255), x=label_x, y=y
        ).draw()
        y -= 32

        # World stats
        snap = self.simulation.get_snapshot()
        text.Label(
            "WORLD", font_name="Segoe UI", font_size=13,
            color=(*MUTED, 255), x=label_x, y=y
        ).draw()
        y -= 22

        stats = [
            f"Agents: {snap['agents']} / {self.cfg.max_agents}",
            f"Avg health: {snap.get('avg_health', 0.0):.1f}",
            f"Avg food: {snap.get('avg_food_level', 0.0):.1f}",
            f"Food: {snap['food']} / {self.cfg.max_food}",
            f"Plants: {snap.get('plants', 0)}",
            f"Meat: {snap.get('meat', 0)}",
            f"Rocks: {snap.get('blocks', 0)}",
            f"Toxic pockets: {snap.get('toxics', 0)}",
            f"Signals: {snap.get('signals', 0)}",
            f"Births: {snap['births']}",
            f"Deaths: {snap['deaths']}",
            f"Eaten: {snap['food_eaten']}",
            f"Terrain coll.: {snap.get('terrain_collisions', 0)}",
            f"Terrain push: {snap.get('terrain_pushes', 0)}",
            f"Toxic ticks: {snap.get('toxic_ticks', 0)}",
            f"Herbivores: {snap.get('herbivores', 0)}",
            f"Carnivores: {snap.get('carnivores', 0)}",
            f"Omnivores: {snap.get('omnivores', 0)}",
            f"Paused: {self.simulation.paused}",
            f"Best offspring: {snap.get('best_offspring', 0)}",
            f"Avg offspring:  {snap.get('avg_offspring', 0):.2f}",
            f"Behavior div: {snap.get('avg_behavior_diversity', 0.0):.3f}",
            f"Avg diet pref: {snap.get('avg_diet_preference', 0.0):+.2f}",
            f"Seed: {self.simulation.current_seed}",
            f"Debug overlays: {debug_overlays}",
            f"Mut Phys: {snap.get('mut_physical', 0):.3f}   [ ]",
            f"Mut Brain: {snap.get('mut_brain', 0):.3f}   ; '",
            f"Speed: {snap.get('time_scale', 1.0):.2f}x   +/-",
        ]
        text.Label(
            f"Tick: {snap['tick']}", font_name="Segoe UI", font_size=13,
            color=(*TEXT_COLOR, 255), x=label_x, y=y
        ).draw()
        y -= 18

        if self.show_run_data:
            text.Label(
                "RUN DATA", font_name="Segoe UI", font_size=13,
                color=(*MUTED, 255), x=label_x, y=y
            ).draw()
            y -= 20

            for s in stats:
                text.Label(s, font_name="Segoe UI", font_size=13,
                           color=(*TEXT_COLOR, 255), x=label_x, y=y).draw()
                y -= 18
        else:
            text.Label(
                "V = Show run data", font_name="Segoe UI", font_size=11,
                color=(*MUTED, 200), x=label_x, y=y
            ).draw()
            y -= 16

        y -= 14

        # Best of Run (very useful for seeing what strategies are winning)
        text.Label(
            "BEST OF RUN", font_name="Segoe UI", font_size=13,
            color=(*MUTED, 255), x=label_x, y=y
        ).draw()
        y -= 20

        if self.simulation.agents:
            # Primary sort is now reproductive success (offspring count)
            top = sorted(
                self.simulation.agents,
                key=lambda a: (a.offspring_count, a.food_eaten, a.age),
                reverse=True
            )[:4]

            for i, agent in enumerate(top, 1):
                text.Label(
                    f"{i}. #{agent.id} ({agent.lineage}) — {agent.offspring_count} offspring",
                    font_name="Segoe UI", font_size=12,
                    color=(*TEXT_COLOR, 255), x=label_x, y=y
                ).draw()
                y -= 16
        else:
            text.Label("No agents alive", font_name="Segoe UI", font_size=12,
                       color=(*MUTED, 255), x=label_x, y=y).draw()
            y -= 16

        y -= 12

        # Selected agent
        text.Label(
            "SELECTED AGENT", font_name="Segoe UI", font_size=13,
            color=(*MUTED, 255), x=label_x, y=y
        ).draw()
        y -= 20

        sel = self._get_selected_entity()
        if sel:
            selected_kind, selected_obj = sel
            text.Label(
                f"TYPE: {selected_kind.upper()}", font_name="Segoe UI", font_size=12,
                color=(*ACCENT, 255), x=label_x, y=y
            ).draw()
            y -= 18

            lines: list[str]
            if selected_kind == "agent":
                agent = selected_obj
                lines = [
                    f"ID: {agent.id}",
                    f"Lineage: {agent.lineage}",
                    f"Age: {agent.age}",
                    f"Energy: {agent.energy:.1f}",
                    f"Health: {agent.health:.1f}",
                    f"Food store: {agent.food_level:.1f}",
                    f"Food eaten: {agent.food_eaten}",
                    f"Diet: {self.simulation._diet_label(agent)}",
                    f"Diet pref: {agent.genome.genes.diet_preference:+.2f}",
                    f"Size: {agent.genome.genes.size:.2f}",
                    f"Plants eaten: {agent.plant_eaten}",
                    f"Meat eaten: {agent.meat_eaten}",
                    f"Offspring:  {agent.offspring_count}",
                    f"Exploration: {agent.genome.genes.exploration_noise:.2f}",
                    f"Hunger sens.: {agent.genome.genes.hunger_sensitivity:.2f}",
                    f"Memory infl.: {agent.genome.genes.memory_influence:.2f}",
                    f"Velocity: {math.hypot(agent.vx, agent.vy):.2f}",
                    f"Toxic exposure: {agent.toxic_exposure:.2f}",
                ]
            elif selected_kind == "food":
                food = selected_obj
                lines = [
                    f"Kind: {food.kind}",
                    f"Energy: {food.energy:.1f}",
                    f"Radius: {food.radius:.1f}",
                    f"Position: {food.x:.1f}, {food.y:.1f}",
                ]
            elif selected_kind == "block":
                block = selected_obj
                lines = [
                    f"Kind: {block.kind}",
                    f"Radius: {block.radius:.1f}",
                    f"Mass: {block.mass:.1f}",
                    f"Habitat bonus: {block.habitat_bonus:.2f}",
                    f"Position: {block.x:.1f}, {block.y:.1f}",
                ]
            else:
                zone = selected_obj
                lines = [
                    f"Radius: {zone.radius:.1f}",
                    f"Strength: {zone.strength:.2f}",
                    f"Position: {zone.x:.1f}, {zone.y:.1f}",
                ]

            for line in lines:
                text.Label(line, font_name="Segoe UI", font_size=13,
                           color=(*TEXT_COLOR, 255), x=label_x, y=y).draw()
                y -= 17

            if selected_kind == "agent":
                signal_snapshot = self.simulation.get_signal_snapshot(selected_obj)
                text.Label(
                    f"Signals heard: food {signal_snapshot['food']:.2f}   danger {signal_snapshot['danger']:.2f}",
                    font_name="Segoe UI", font_size=11,
                    color=(200, 220, 255, 255), x=label_x, y=y
                ).draw()
                y -= 16

                # Explicit memory readout (proof of recurrent memory implementation)
                if self.cfg.memory_size > 0:
                    mem_str = ", ".join(f"{v:+.2f}" for v in selected_obj.memory[:6])
                    text.Label(
                        f"Memory: [{mem_str}]",
                        font_name="Segoe UI", font_size=11,
                        color=(180, 220, 255), x=label_x, y=y
                    ).draw()
                    y -= 16

                # === See the agent's "thoughts" ===
                y -= 10
                text.Label(
                    "BRAIN STATE — Hidden Neurons (0-7)",
                    font_name="Segoe UI", font_size=11,
                    color=(*MUTED, 255), x=label_x, y=y
                ).draw()
                y -= 18

                try:
                    sensors = self.simulation._get_sensor_inputs(selected_obj)
                    diag = selected_obj.genome.get_diagnostics(sensors)
                    activations = diag["hidden_activations"]
                    importance = diag["input_importance"]

                    # Hidden layer bars
                    bar_width = 18
                    bar_height = 24
                    start_x = label_x

                    for i, act in enumerate(activations):
                        intensity = (act + 1.0) / 2.0
                        height = max(2, int(bar_height * intensity))
                        r = int(140 + 115 * intensity)
                        g = int(160 + 95 * intensity)
                        b = int(255 - 210 * intensity)

                        bar_x = start_x + i * (bar_width + 2)
                        bar_y = y - bar_height
                        shapes.Rectangle(bar_x, bar_y, bar_width, bar_height, color=(30, 35, 45)).draw()
                        shapes.Rectangle(bar_x, bar_y, bar_width, height, color=(r, g, b)).draw()

                    y -= 4

                    # Current decisions
                    text.Label(
                        f"Decisions → Turn: {diag['turn']:+.2f}   Thrust: {diag['thrust']:.2f}",
                        font_name="Segoe UI", font_size=11,
                        color=(*ACCENT, 255), x=label_x, y=y
                    ).draw()
                    y -= 16

                    # Memory state (recurrent memory - within-lifetime state)
                    if self.cfg.memory_size > 0 and "new_memory" in diag and len(diag["new_memory"]) > 0:
                        mem = diag["new_memory"]
                        y -= 6
                        text.Label(
                            "MEMORY (6 recurrent units)",
                            font_name="Segoe UI", font_size=10,
                            color=(*MUTED, 255), x=label_x, y=y
                        ).draw()
                        y -= 15

                        mem_bar_width = 24
                        for i, m in enumerate(mem):
                            intensity = (m + 1.0) / 2.0
                            height = max(2, int(16 * intensity))
                            r = int(90 + 130 * intensity)
                            g = int(170 + 70 * intensity)
                            b = int(210 + 45 * intensity)

                            bar_x = label_x + i * (mem_bar_width + 2)
                            shapes.Rectangle(bar_x, y - 16, mem_bar_width, 16, color=(28, 32, 42)).draw()
                            shapes.Rectangle(bar_x, y - 16, mem_bar_width, height, color=(r, g, b)).draw()

                        y -= 6

                    # Input saliency — which sensors matter most right now?
                    ray_labels = [f"R{i}" for i in range(8)] + ["Energy", "Speed", "Food", "Health", "Size", "Threat", "Toxicity", "Bias", "Food Sig", "Danger Sig"]
                    top_indices = np.argsort(importance)[-3:][::-1]

                    top_str = ", ".join(
                        f"{ray_labels[i]}({importance[i]:.1f})" for i in top_indices
                    )
                    text.Label(
                        f"Paying attention to: {top_str}",
                        font_name="Segoe UI", font_size=10,
                        color=(200, 210, 180), x=label_x, y=y
                    ).draw()
                    y -= 16

                except Exception:
                    text.Label("(brain state unavailable)", font_name="Segoe UI", font_size=11,
                               color=(*MUTED, 255), x=label_x, y=y).draw()
                    y -= 14

        else:
            text.Label("Click an agent to inspect", font_name="Segoe UI", font_size=13,
                       color=(*MUTED, 255), x=label_x, y=y).draw()
            y -= 14

        y -= 18   # more breathing room before help text

        # Help text: anchor near the bottom of the (now taller) sidebar
        if self.show_help:
            help_y = self.height - 260   # start help ~260px from bottom of window
            y = min(y - 10, help_y)

            text.Label(
                "CONTROLS", font_name="Segoe UI", font_size=13,
                color=(*MUTED, 255), x=label_x, y=y
            ).draw()
            y -= 16

            help_lines = [
                "Space   Pause / resume",
                "+/-     Time speed",
                "Mouse Wheel   Zoom",
                "R       Reset (with elites)",
                "S       Print run summary",
                "P       Plot last run (matplotlib)",
                "V       Toggle run data",
                "[ ]     Phys mutation rate",
                "; '     Brain mutation rate",
                "Click   Select agent",
                "H       Hide controls",
                "Esc     Quit",
            ]
            for line in help_lines:
                text.Label(line, font_name="Segoe UI", font_size=12,
                           color=(*TEXT_COLOR, 255), x=label_x, y=y).draw()
                y -= 16
        else:
            # Always show a hint when help is hidden
            text.Label("H = Show controls   |   V = Toggle run data   Wheel = Zoom   R=reset  S=summary  P=plot  +/- = speed  [ ] ; ' = mutation", font_name="Segoe UI", font_size=11,
                       color=(*MUTED, 200), x=label_x, y=22).draw()

    def _lineage_color(self, lineage: str) -> tuple[int, int, int]:
        index = sum(ord(char) for char in lineage) % len(LINEAGE_PALETTE)
        return LINEAGE_PALETTE[index]

    def _get_selected_entity(self) -> tuple[str, object] | None:
        if self.selected_entity is None:
            return None

        kind, item = self.selected_entity
        if kind == "agent" and item in self.simulation.agents:
            return self.selected_entity
        if kind == "food" and item in self.simulation.foods:
            return self.selected_entity
        if kind == "block" and item in self.simulation.blocks:
            return self.selected_entity
        if kind == "zone" and item in self.simulation.toxic_zones:
            return self.selected_entity
        return None

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
        elif symbol == key.SPACE:
            self.simulation.paused = not self.simulation.paused
        elif symbol == key.R:
            # Reset with inter-run elitism using both reproductive success + behavioral diversity
            viable = [a for a in self.simulation.agents if a.age > 150]

            if viable:
                elites = self.simulation._select_diverse_elites(viable, 6)
            else:
                elites = sorted(
                    self.simulation.agents,
                    key=lambda a: (a.offspring_count, a.food_eaten, a.age),
                    reverse=True
                )[:4]

            elite_genomes = [a.genome.copy() for a in elites]
            self.simulation.reset(elite_genomes=elite_genomes)
            self.selected_entity = None

        elif symbol == key.S:
            # Print detailed summary + save JSON log (no reset)
            self.simulation._print_run_summary()
            self.simulation._save_run_log()

        elif symbol == key.P:
            # Export plots from the last run using matplotlib
            self.simulation.plot_last_run()
        elif symbol == key.H:
            self.show_help = not self.show_help
        elif symbol == key.V:
            self.show_run_data = not self.show_run_data

        # Mutation rate god-mode controls
        elif symbol == key.BRACKETLEFT:
            self.simulation.adjust_mutation_physical(-0.005)
        elif symbol == key.BRACKETRIGHT:
            self.simulation.adjust_mutation_physical(+0.005)
        elif symbol == key.SEMICOLON:
            self.simulation.adjust_mutation_brain(-0.005)
        elif symbol == key.APOSTROPHE:
            self.simulation.adjust_mutation_brain(+0.005)

        # Time scaling (smooth +/-)
        elif symbol == key.PLUS or symbol == key.EQUAL:
            self.simulation.adjust_time_scale(1.25)
        elif symbol == key.MINUS or symbol == key.UNDERSCORE:
            self.simulation.adjust_time_scale(0.8)

    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            vp_x, vp_y, vp_w, vp_h = self._get_world_viewport()
            if vp_x <= x < vp_x + vp_w and vp_y <= y < vp_y + vp_h:
                self._camera_drag_start = (x, y)
                self._camera_drag_origin = (self.camera_x, self.camera_y)
                self._camera_dragging = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not (buttons & mouse.LEFT):
            return

        if self._camera_drag_start is None or self._camera_drag_origin is None:
            return

        vp_x, vp_y, vp_w, vp_h = self._get_world_viewport()
        if not (vp_x <= x < vp_x + vp_w and vp_y <= y < vp_y + vp_h):
            return

        drag_dx = x - self._camera_drag_start[0]
        drag_dy = y - self._camera_drag_start[1]
        if not self._camera_dragging and math.hypot(drag_dx, drag_dy) < self._camera_drag_threshold:
            return

        self._camera_dragging = True

        scale = vp_w / self.cfg.world_width
        effective_scale = scale * self.zoom

        origin_x, origin_y = self._camera_drag_origin
        self.camera_x = origin_x - drag_dx / effective_scale
        self.camera_y = origin_y - drag_dy / effective_scale

        max_cam_x = max(0.0, self.cfg.world_width - (self.cfg.world_width / self.zoom))
        max_cam_y = max(0.0, self.cfg.world_height - (self.cfg.world_height / self.zoom))
        self.camera_x = max(0.0, min(self.camera_x, max_cam_x))
        self.camera_y = max(0.0, min(self.camera_y, max_cam_y))

    def on_mouse_release(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return

        vp_x, vp_y, vp_w, vp_h = self._get_world_viewport()
        was_dragging = self._camera_dragging
        self._camera_dragging = False

        if self._camera_drag_start is not None and not was_dragging:
            if vp_x <= x < vp_x + vp_w and vp_y <= y < vp_y + vp_h:
                wx, wy = self._screen_to_world(x, y)
                self.selected_entity = self._find_entity_at(wx, wy)

        self._camera_drag_start = None
        self._camera_drag_origin = None

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        vp_x, vp_y, vp_w, vp_h = self._get_world_viewport()
        if not (vp_x <= x < vp_x + vp_w and vp_y <= y < vp_y + vp_h) or scroll_y == 0:
            return

        wx, wy = self._screen_to_world(x, y)

        factor = 1.12 if scroll_y > 0 else (1.0 / 1.12)
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))

        if abs(new_zoom - self.zoom) < 0.001:
            return

        # Cursor-centered zoom using current viewport
        scale = vp_w / self.cfg.world_width
        self.camera_x = wx - (x - vp_x) / (scale * new_zoom)
        self.camera_y = wy - (y - vp_y) / (scale * new_zoom)

        self.zoom = new_zoom

        # Soft camera clamp
        max_cam_x = max(0.0, self.cfg.world_width - (self.cfg.world_width / self.zoom))
        max_cam_y = max(0.0, self.cfg.world_height - (self.cfg.world_height / self.zoom))
        self.camera_x = max(0.0, min(self.camera_x, max_cam_x))
        self.camera_y = max(0.0, min(self.camera_y, max_cam_y))

    def _screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert screen coordinates to world coordinates.
        Returns (0,0) if the point is outside the world viewport.
        """
        vp_x, vp_y, vp_w, vp_h = self._get_world_viewport()

        if not (vp_x <= sx < vp_x + vp_w and vp_y <= sy < vp_y + vp_h):
            return 0.0, 0.0

        scale = vp_w / self.cfg.world_width   # uniform because we maintain aspect
        effective_scale = scale * self.zoom

        wx = self.camera_x + (sx - vp_x) / effective_scale
        wy = self.camera_y + (sy - vp_y) / effective_scale
        return wx, wy

    def _find_entity_at(self, mx: float, my: float) -> tuple[str, object] | None:
        best_kind: str | None = None
        best_item: object | None = None
        best_dist = 9999.0

        for agent in self.simulation.agents:
            d = math.hypot(agent.x - mx, agent.y - my)
            threshold = self.cfg.agent_radius * (0.82 + agent.genome.genes.size * 0.88) + 8.0
            if d < best_dist and d < threshold:
                best_dist = d
                best_kind = "agent"
                best_item = agent

        for food in self.simulation.foods:
            d = math.hypot(food.x - mx, food.y - my)
            threshold = food.radius + 7.0
            if d < best_dist and d < threshold:
                best_dist = d
                best_kind = "food"
                best_item = food

        for block in self.simulation.blocks:
            d = math.hypot(block.x - mx, block.y - my)
            threshold = block.radius + 10.0
            if d < best_dist and d < threshold:
                best_dist = d
                best_kind = "block"
                best_item = block

        for zone in self.simulation.toxic_zones:
            d = math.hypot(zone.x - mx, zone.y - my)
            threshold = zone.radius
            if d < best_dist and d < threshold:
                best_dist = d
                best_kind = "zone"
                best_item = zone

        if best_kind is None or best_item is None:
            return None
        return best_kind, best_item

    def _get_world_viewport(self) -> tuple[int, int, int, int]:
        """Returns (x, y, w, h) in screen pixels for the world rendering area.
        Always maintains the world's fixed aspect ratio (letterbox/pillarbox as needed).
        The world is treated as a fixed 1280x720 logical space.
        """
        if self._world_viewport is not None:
            return self._world_viewport

        sidebar_w = 320
        avail_w = max(1, self.width - sidebar_w)
        avail_h = max(1, self.height)

        world_w = float(self.cfg.world_width)
        world_h = float(self.cfg.world_height)
        world_aspect = world_w / world_h

        if avail_w / avail_h > world_aspect:
            # Pillarbox (taller window)
            vp_h = avail_h
            vp_w = int(vp_h * world_aspect)
            vp_x = (avail_w - vp_w) // 2
            vp_y = 0
        else:
            # Letterbox (wider window)
            vp_w = avail_w
            vp_h = int(vp_w / world_aspect)
            vp_x = 0
            vp_y = (avail_h - vp_h) // 2

        self._world_viewport = (vp_x, vp_y, vp_w, vp_h)
        return self._world_viewport

    def _invalidate_viewport(self):
        self._world_viewport = None

    def on_resize(self, width: int, height: int):
        self._invalidate_viewport()

    def on_close(self):
        pyglet.app.exit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="primordia")
    parser.add_argument("--experiment", action="store_true", help="run parallel headless worlds instead of the UI")
    parser.add_argument("--preset", type=str, default=None, help="named experiment preset to use")
    parser.add_argument("--compare", nargs="+", default=None, help="run and compare multiple presets")
    parser.add_argument("--list-presets", action="store_true", help="list available experiment presets")
    parser.add_argument("--worlds", type=int, default=8, help="number of worlds to run in parallel")
    parser.add_argument("--ticks", type=int, default=6000, help="ticks per world for experiment mode")
    parser.add_argument("--workers", type=int, default=None, help="process worker count for experiment mode")
    parser.add_argument("--seed", type=int, default=None, help="base seed for experiment mode")
    args = parser.parse_args(argv)

    if args.list_presets:
        print("Available presets:")
        for name in list_experiment_presets():
            print(f"  - {name}")
        return

    if args.experiment:
        if args.compare:
            run_preset_comparison(
                presets=args.compare,
                worlds=args.worlds,
                ticks_per_world=args.ticks,
                workers=args.workers,
                base_seed=args.seed,
            )
        else:
            run_parallel_experiment(
                worlds=args.worlds,
                ticks_per_world=args.ticks,
                workers=args.workers,
                base_seed=args.seed,
                preset=args.preset,
            )
        return

    window = PrimordiaWindow()
    pyglet.app.run()


if __name__ == "__main__":
    main()
