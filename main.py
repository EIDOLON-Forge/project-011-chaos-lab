"""
EIDOLON #011 — CHAOS LAB
=========================

Interactive 3D Lorenz Attractor visualization.

The Lorenz system:

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Default parameters:

    sigma = 10
    rho   = 28
    beta  = 8 / 3

Requirements:
    Python 3.12+
    Pygame 2.6.1

Run:
    python main.py

Smoke test:
    python main.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass

import pygame


# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 1200
HEIGHT = 800
FPS = 60

HEADER_HEIGHT = 76
SIDEBAR_WIDTH = 300
MARGIN = 18

SIMULATION_AREA = pygame.Rect(
    MARGIN,
    HEADER_HEIGHT + MARGIN,
    WIDTH - SIDEBAR_WIDTH - MARGIN * 3,
    HEIGHT - HEADER_HEIGHT - MARGIN * 2,
)

MAX_TRAIL_POINTS = 18000
DEFAULT_TRAIL_POINTS = 10000

DEFAULT_DT = 0.008

CAMERA_DISTANCE = 145.0
CAMERA_FOV = 620.0

DEFAULT_YAW = 0.55
DEFAULT_PITCH = -0.28

POINT_SCALE = 2.4

INTRO_DURATION = 2.0
COUNTDOWN_STEP = 0.75


# ============================================================
# COLOURS
# ============================================================

BLACK = (3, 5, 8)
PANEL = (9, 13, 18)
PANEL_LIGHT = (15, 21, 28)
GRID = (25, 35, 44)

WHITE = (238, 245, 248)
MUTED = (125, 142, 153)
DIM = (70, 84, 94)

CYAN = (80, 220, 220)
TEAL = (40, 170, 170)
GREEN = (100, 230, 150)
AMBER = (240, 190, 80)
VIOLET = (180, 120, 240)
PINK = (245, 100, 170)
BLUE = (90, 150, 255)
RED = (245, 90, 90)


PALETTES = (
    (
        "EIDOLON",
        (
            (70, 255, 230),
            (130, 245, 220),
            (220, 255, 245),
        ),
    ),
    (
        "FIRE",
        (
            (255, 75, 30),
            (255, 160, 40),
            (255, 240, 150),
        ),
    ),
    (
        "OCEAN",
        (
            (40, 130, 255),
            (80, 210, 255),
            (190, 250, 255),
        ),
    ),
    (
        "VIOLET",
        (
            (120, 70, 240),
            (190, 100, 255),
            (245, 190, 255),
        ),
    ),
    (
        "MATRIX",
        (
            (20, 160, 70),
            (70, 230, 110),
            (190, 255, 210),
        ),
    ),
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Point3D:
    x: float
    y: float
    z: float


@dataclass
class ProjectedPoint:
    x: float
    y: float
    depth: float


@dataclass
class LorenzParameters:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


@dataclass
class Camera:
    yaw: float = DEFAULT_YAW
    pitch: float = DEFAULT_PITCH
    distance: float = CAMERA_DISTANCE
    auto_rotate: bool = False


@dataclass
class Particle:
    position: Point3D
    velocity: Point3D
    trail: list[Point3D]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


def lerp(
    a: float,
    b: float,
    amount: float,
) -> float:
    return a + (b - a) * amount


def lerp_colour(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    amount = clamp(amount, 0.0, 1.0)

    return (
        int(lerp(a[0], b[0], amount)),
        int(lerp(a[1], b[1], amount)),
        int(lerp(a[2], b[2], amount)),
    )


def palette_colour(
    palette: tuple[tuple[int, int, int], ...],
    amount: float,
) -> tuple[int, int, int]:
    if len(palette) == 1:
        return palette[0]

    amount %= 1.0

    position = amount * (len(palette) - 1)
    index = int(position)
    fraction = position - index

    if index >= len(palette) - 1:
        return palette[-1]

    return lerp_colour(
        palette[index],
        palette[index + 1],
        fraction,
    )


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


# ============================================================
# LORENZ MATHEMATICS
# ============================================================

def lorenz_derivatives(
    point: Point3D,
    parameters: LorenzParameters,
) -> Point3D:
    dx = parameters.sigma * (point.y - point.x)

    dy = (
        point.x
        * (parameters.rho - point.z)
        - point.y
    )

    dz = (
        point.x * point.y
        - parameters.beta * point.z
    )

    return Point3D(dx, dy, dz)


def rk4_step(
    point: Point3D,
    dt: float,
    parameters: LorenzParameters,
) -> Point3D:
    """
    Fourth-order Runge-Kutta integration.

    This is more stable and accurate than simple Euler integration
    while remaining inexpensive enough for real-time visualization.
    """

    k1 = lorenz_derivatives(
        point,
        parameters,
    )

    p2 = Point3D(
        point.x + k1.x * dt * 0.5,
        point.y + k1.y * dt * 0.5,
        point.z + k1.z * dt * 0.5,
    )

    k2 = lorenz_derivatives(
        p2,
        parameters,
    )

    p3 = Point3D(
        point.x + k2.x * dt * 0.5,
        point.y + k2.y * dt * 0.5,
        point.z + k2.z * dt * 0.5,
    )

    k3 = lorenz_derivatives(
        p3,
        parameters,
    )

    p4 = Point3D(
        point.x + k3.x * dt,
        point.y + k3.y * dt,
        point.z + k3.z * dt,
    )

    k4 = lorenz_derivatives(
        p4,
        parameters,
    )

    return Point3D(
        point.x
        + dt
        * (
            k1.x
            + 2.0 * k2.x
            + 2.0 * k3.x
            + k4.x
        )
        / 6.0,

        point.y
        + dt
        * (
            k1.y
            + 2.0 * k2.y
            + 2.0 * k3.y
            + k4.y
        )
        / 6.0,

        point.z
        + dt
        * (
            k1.z
            + 2.0 * k2.z
            + 2.0 * k3.z
            + k4.z
        )
        / 6.0,
    )


# ============================================================
# LORENZ SYSTEM
# ============================================================

class LorenzSystem:

    def __init__(
        self,
        parameters: LorenzParameters,
    ) -> None:
        self.parameters = parameters

        self.particles: list[Particle] = []

        self.trail_limit = DEFAULT_TRAIL_POINTS

        self.speed = 1.0

        self.running = True

        self.total_steps = 0

        self.reset()

    def create_particle(
        self,
        offset: float = 0.0,
    ) -> Particle:

        base_x = 0.01 + offset
        base_y = 0.0 + offset
        base_z = 0.0

        position = Point3D(
            base_x,
            base_y,
            base_z,
        )

        velocity = lorenz_derivatives(
            position,
            self.parameters,
        )

        return Particle(
            position=position,
            velocity=velocity,
            trail=[position],
        )

    def reset(
        self,
        particle_count: int = 1,
    ) -> None:

        self.particles.clear()

        for index in range(particle_count):

            offset = index * 0.0008

            particle = self.create_particle(
                offset=offset,
            )

            self.particles.append(particle)

        self.total_steps = 0

    def randomize(
        self,
        particle_count: int = 1,
    ) -> None:

        self.particles.clear()

        for _ in range(particle_count):

            position = Point3D(
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(0.0, 1.0),
            )

            velocity = lorenz_derivatives(
                position,
                self.parameters,
            )

            self.particles.append(
                Particle(
                    position=position,
                    velocity=velocity,
                    trail=[position],
                )
            )

        self.total_steps = 0

    def step(self) -> None:

        for particle in self.particles:

            new_position = rk4_step(
                particle.position,
                DEFAULT_DT * self.speed,
                self.parameters,
            )

            particle.position = new_position

            particle.velocity = lorenz_derivatives(
                new_position,
                self.parameters,
            )

            particle.trail.append(
                Point3D(
                    new_position.x,
                    new_position.y,
                    new_position.z,
                )
            )

            if len(particle.trail) > self.trail_limit:
                del particle.trail[
                    :len(particle.trail)
                    - self.trail_limit
                ]

        self.total_steps += 1

    def update(
        self,
        steps: int,
    ) -> None:

        if not self.running:
            return

        for _ in range(steps):
            self.step()

    def set_particle_count(
        self,
        count: int,
    ) -> None:

        count = int(
            clamp(count, 1, 8)
        )

        if count == len(self.particles):
            return

        if count > len(self.particles):

            while len(self.particles) < count:

                index = len(self.particles)

                offset = index * 0.0008

                self.particles.append(
                    self.create_particle(offset)
                )

        else:

            self.particles = (
                self.particles[:count]
            )

    def clear_trails(self) -> None:

        for particle in self.particles:

            particle.trail = [
                Point3D(
                    particle.position.x,
                    particle.position.y,
                    particle.position.z,
                )
            ]


# ============================================================
# CAMERA
# ============================================================

class CameraController:

    def __init__(self) -> None:
        self.camera = Camera()

    def rotate(
        self,
        yaw_delta: float,
        pitch_delta: float,
    ) -> None:

        self.camera.yaw += yaw_delta

        self.camera.pitch = clamp(
            self.camera.pitch + pitch_delta,
            -1.35,
            1.35,
        )

    def zoom(
        self,
        amount: float,
    ) -> None:

        self.camera.distance = clamp(
            self.camera.distance + amount,
            55.0,
            280.0,
        )

    def reset(self) -> None:
        self.camera = Camera()

    def transform(
        self,
        point: Point3D,
    ) -> Point3D:

        yaw = self.camera.yaw
        pitch = self.camera.pitch

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        x = (
            point.x * cos_yaw
            - point.z * sin_yaw
        )

        z = (
            point.x * sin_yaw
            + point.z * cos_yaw
        )

        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)

        y = (
            point.y * cos_pitch
            - z * sin_pitch
        )

        depth = (
            point.y * sin_pitch
            + z * cos_pitch
        )

        return Point3D(
            x,
            y,
            depth,
        )

    def project(
        self,
        point: Point3D,
        rect: pygame.Rect,
    ) -> ProjectedPoint | None:

        transformed = self.transform(point)

        depth = (
            transformed.z
            + self.camera.distance
        )

        if depth <= 1.0:
            return None

        factor = (
            CAMERA_FOV
            / depth
        )

        screen_x = (
            rect.centerx
            + transformed.x * factor
        )

        screen_y = (
            rect.centery
            - transformed.y * factor
        )

        return ProjectedPoint(
            screen_x,
            screen_y,
            depth,
        )


# ============================================================
# AUDIO
# ============================================================

class AudioManager:

    def __init__(self) -> None:

        self.enabled = True

        self.available = False

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=44100,
                    size=-16,
                    channels=1,
                    buffer=512,
                )

            self.available = True

        except pygame.error:
            self.available = False

    def tone(
        self,
        frequency: float,
        duration: float = 0.08,
        volume: float = 0.08,
    ) -> None:

        if not self.enabled or not self.available:
            return

        sample_rate = 44100

        samples = max(
            1,
            int(sample_rate * duration),
        )

        buffer = bytearray()

        for index in range(samples):

            t = index / sample_rate

            envelope = min(
                1.0,
                t / 0.01,
            )

            envelope *= min(
                1.0,
                (duration - t) / 0.02,
            )

            sample = math.sin(
                2.0
                * math.pi
                * frequency
                * t
            )

            value = int(
                sample
                * envelope
                * volume
                * 32767
            )

            buffer += int(value).to_bytes(
                2,
                byteorder="little",
                signed=True,
            )

        try:

            sound = pygame.mixer.Sound(
                buffer=bytes(buffer)
            )

            sound.play()

        except pygame.error:
            pass

    def toggle(self) -> None:

        self.enabled = not self.enabled

        if self.enabled:
            self.tone(520.0)


# ============================================================
# UI
# ============================================================

class UI:

    def __init__(self) -> None:

        self.fonts = {
            "title": pygame.font.SysFont(
                "consolas",
                30,
                bold=True,
            ),

            "large": pygame.font.SysFont(
                "consolas",
                24,
                bold=True,
            ),

            "normal": pygame.font.SysFont(
                "consolas",
                17,
            ),

            "small": pygame.font.SysFont(
                "consolas",
                14,
            ),

            "tiny": pygame.font.SysFont(
                "consolas",
                12,
            ),
        }

    def text(
        self,
        surface: pygame.Surface,
        value: str,
        position: tuple[int, int],
        font: str = "normal",
        colour: tuple[int, int, int] = WHITE,
        anchor: str = "topleft",
    ) -> pygame.Rect:

        rendered = self.fonts[font].render(
            value,
            True,
            colour,
        )

        rect = rendered.get_rect()

        setattr(
            rect,
            anchor,
            position,
        )

        surface.blit(
            rendered,
            rect,
        )

        return rect

    def panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
    ) -> None:

        pygame.draw.rect(
            surface,
            PANEL,
            rect,
            border_radius=8,
        )

        pygame.draw.rect(
            surface,
            GRID,
            rect,
            width=1,
            border_radius=8,
        )

    def line(
        self,
        surface: pygame.Surface,
        start: tuple[int, int],
        end: tuple[int, int],
        colour: tuple[int, int, int],
        width: int = 1,
    ) -> None:

        pygame.draw.line(
            surface,
            colour,
            start,
            end,
            width,
        )


# ============================================================
# APPLICATION
# ============================================================

class ChaosLab:

    def __init__(
        self,
        smoke_test: bool = False,
    ) -> None:

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "EIDOLON #011 — CHAOS LAB"
        )

        self.clock = pygame.time.Clock()

        self.ui = UI()

        self.audio = AudioManager()

        self.parameters = LorenzParameters()

        self.system = LorenzSystem(
            self.parameters
        )

        self.camera = CameraController()

        self.palette_index = 0

        self.palette_shift = 0.0

        self.intensity = 1.0

        self.glow_enabled = True

        self.grid_enabled = True

        self.show_hud = True

        self.show_axes = True

        self.intro_timer = INTRO_DURATION

        self.countdown = 3

        self.countdown_timer = 0.0

        self.started = False

        self.running = True

        self.frame_time = 0.0

        self.fps = 0.0

        self.mouse_dragging = False

        self.last_mouse = None

        self.screenshot_index = 1

        self.smoke_test = smoke_test

        if smoke_test:
            self.run_smoke_test()

    # --------------------------------------------------------
    # PROPERTIES
    # --------------------------------------------------------

    @property
    def palette_name(self) -> str:
        return PALETTES[
            self.palette_index
        ][0]

    @property
    def palette(
        self,
    ) -> tuple[tuple[int, int, int], ...]:
        return PALETTES[
            self.palette_index
        ][1]

    # --------------------------------------------------------
    # SMOKE TEST
    # --------------------------------------------------------

    def run_smoke_test(self) -> None:

        self.system.reset(1)

        for _ in range(100):
            self.system.step()

        if not self.system.particles:
            raise RuntimeError(
                "Lorenz smoke test failed."
            )

        point = (
            self.system.particles[0]
            .position
        )

        if not all(
            math.isfinite(value)
            for value in (
                point.x,
                point.y,
                point.z,
            )
        ):
            raise RuntimeError(
                "Lorenz simulation produced "
                "a non-finite value."
            )

        print(
            "Chaos Lab smoke test: OK"
        )

        print(
            "Lorenz parameters:",
            f"sigma={self.parameters.sigma},",
            f"rho={self.parameters.rho},",
            f"beta={self.parameters.beta:.6f}",
        )

        print(
            "Trail points:",
            len(
                self.system.particles[0].trail
            ),
        )

        raise SystemExit(0)

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self) -> None:

        self.system.reset(
            len(self.system.particles)
        )

        self.camera.reset()

        self.parameters.sigma = 10.0
        self.parameters.rho = 28.0
        self.parameters.beta = 8.0 / 3.0

        self.palette_shift = 0.0

        self.audio.tone(
            300.0,
            0.08,
        )

    # --------------------------------------------------------
    # PALETTE
    # --------------------------------------------------------

    def next_palette(self) -> None:

        self.palette_index = (
            self.palette_index + 1
        ) % len(PALETTES)

        self.audio.tone(
            420.0 + self.palette_index * 70.0
        )

    # --------------------------------------------------------
    # SCREENSHOT
    # --------------------------------------------------------

    def screenshot(self) -> None:

        filename = (
            f"chaos_lab_"
            f"{self.screenshot_index:03d}.png"
        )

        pygame.image.save(
            self.screen,
            filename,
        )

        self.screenshot_index += 1

        self.audio.tone(
            800.0,
            0.07,
        )

        print(
            f"Screenshot saved: {filename}"
        )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    def parameter_preset(
        self,
        preset: str,
    ) -> None:

        if preset == "classic":

            self.parameters.sigma = 10.0
            self.parameters.rho = 28.0
            self.parameters.beta = 8.0 / 3.0

        elif preset == "stable":

            self.parameters.sigma = 10.0
            self.parameters.rho = 20.0
            self.parameters.beta = 8.0 / 3.0

        elif preset == "chaotic":

            self.parameters.sigma = 14.0
            self.parameters.rho = 35.0
            self.parameters.beta = 3.0

        self.system.clear_trails()

        self.audio.tone(
            580.0
        )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    def handle_event(
        self,
        event: pygame.event.Event,
    ) -> None:

        if event.type == pygame.QUIT:
            self.running = False
            return

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                self.running = False

            elif event.key == pygame.K_SPACE:

                self.system.running = (
                    not self.system.running
                )

                self.audio.tone(
                    450.0 if self.system.running
                    else 260.0
                )

            elif event.key == pygame.K_r:
                self.reset()

            elif event.key == pygame.K_c:
                self.system.clear_trails()

            elif event.key == pygame.K_p:
                self.next_palette()

            elif event.key == pygame.K_g:
                self.glow_enabled = (
                    not self.glow_enabled
                )

            elif event.key == pygame.K_h:
                self.show_hud = (
                    not self.show_hud
                )

            elif event.key == pygame.K_a:
                self.show_axes = (
                    not self.show_axes
                )

            elif event.key == pygame.K_t:
                self.grid_enabled = (
                    not self.grid_enabled
                )

            elif event.key == pygame.K_m:
                self.audio.toggle()

            elif event.key == pygame.K_v:
                self.camera.camera.auto_rotate = (
                    not self.camera.camera.auto_rotate
                )

            elif event.key == pygame.K_j:
                self.system.randomize(
                    len(self.system.particles)
                )

            elif event.key == pygame.K_1:
                self.parameter_preset(
                    "classic"
                )

            elif event.key == pygame.K_2:
                self.parameter_preset(
                    "stable"
                )

            elif event.key == pygame.K_3:
                self.parameter_preset(
                    "chaotic"
                )

            elif event.key == pygame.K_s:
                self.screenshot()

            elif event.key in (
                pygame.K_EQUALS,
                pygame.K_PLUS,
            ):
                self.system.speed = clamp(
                    self.system.speed + 0.1,
                    0.1,
                    4.0,
                )

            elif event.key == pygame.K_MINUS:
                self.system.speed = clamp(
                    self.system.speed - 0.1,
                    0.1,
                    4.0,
                )

            elif event.key == pygame.K_LEFTBRACKET:

                self.system.trail_limit = int(
                    clamp(
                        self.system.trail_limit - 1000,
                        2000,
                        MAX_TRAIL_POINTS,
                    )
                )

            elif event.key == pygame.K_RIGHTBRACKET:

                self.system.trail_limit = int(
                    clamp(
                        self.system.trail_limit + 1000,
                        2000,
                        MAX_TRAIL_POINTS,
                    )
                )

            elif event.key == pygame.K_UP:

                self.parameters.rho = clamp(
                    self.parameters.rho + 0.5,
                    5.0,
                    60.0,
                )

                self.system.clear_trails()

            elif event.key == pygame.K_DOWN:

                self.parameters.rho = clamp(
                    self.parameters.rho - 0.5,
                    5.0,
                    60.0,
                )

                self.system.clear_trails()

            elif event.key == pygame.K_RIGHT:

                self.parameters.sigma = clamp(
                    self.parameters.sigma + 0.5,
                    1.0,
                    30.0,
                )

                self.system.clear_trails()

            elif event.key == pygame.K_LEFT:

                self.parameters.sigma = clamp(
                    self.parameters.sigma - 0.5,
                    1.0,
                    30.0,
                )

                self.system.clear_trails()

            elif event.key == pygame.K_z:

                self.parameters.beta = clamp(
                    self.parameters.beta + 0.05,
                    0.5,
                    8.0,
                )

                self.system.clear_trails()

            elif event.key == pygame.K_x:

                self.parameters.beta = clamp(
                    self.parameters.beta - 0.05,
                    0.5,
                    8.0,
                )

                self.system.clear_trails()

        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                self.mouse_dragging = True

                self.last_mouse = event.pos

            elif event.button == 4:

                self.camera.zoom(-8.0)

            elif event.button == 5:

                self.camera.zoom(8.0)

        elif event.type == pygame.MOUSEBUTTONUP:

            if event.button == 1:
                self.mouse_dragging = False
                self.last_mouse = None

        elif event.type == pygame.MOUSEMOTION:

            if (
                self.mouse_dragging
                and self.last_mouse is not None
            ):

                dx = (
                    event.pos[0]
                    - self.last_mouse[0]
                )

                dy = (
                    event.pos[1]
                    - self.last_mouse[1]
                )

                self.camera.rotate(
                    dx * 0.008,
                    dy * 0.006,
                )

                self.last_mouse = event.pos

        elif event.type == pygame.MOUSEWHEEL:

            self.camera.zoom(
                -event.y * 7.0
            )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    def draw_grid(
        self,
        surface: pygame.Surface,
    ) -> None:

        if not self.grid_enabled:
            return

        rect = SIMULATION_AREA

        step = 50

        for x in range(
            rect.left,
            rect.right,
            step,
        ):

            pygame.draw.line(
                surface,
                GRID,
                (x, rect.top),
                (x, rect.bottom),
                1,
            )

        for y in range(
            rect.top,
            rect.bottom,
            step,
        ):

            pygame.draw.line(
                surface,
                GRID,
                (rect.left, y),
                (rect.right, y),
                1,
            )

        pygame.draw.rect(
            surface,
            GRID,
            rect,
            1,
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    def draw_axes(
        self,
        surface: pygame.Surface,
    ) -> None:

        if not self.show_axes:
            return

        origin = Point3D(
            0.0,
            0.0,
            0.0,
        )

        axis_length = 22.0

        axes = (
            (
                Point3D(
                    axis_length,
                    0.0,
                    0.0,
                ),
                RED,
                "X",
            ),
            (
                Point3D(
                    0.0,
                    axis_length,
                    0.0,
                ),
                GREEN,
                "Y",
            ),
            (
                Point3D(
                    0.0,
                    0.0,
                    axis_length,
                ),
                BLUE,
                "Z",
            ),
        )

        origin_projected = self.camera.project(
            origin,
            SIMULATION_AREA,
        )

        if origin_projected is None:
            return

        for endpoint, colour, label in axes:

            projected = self.camera.project(
                endpoint,
                SIMULATION_AREA,
            )

            if projected is None:
                continue

            pygame.draw.line(
                surface,
                colour,
                (
                    int(origin_projected.x),
                    int(origin_projected.y),
                ),
                (
                    int(projected.x),
                    int(projected.y),
                ),
                2,
            )

            self.ui.text(
                surface,
                label,
                (
                    int(projected.x),
                    int(projected.y),
                ),
                "small",
                colour,
            )

    # --------------------------------------------------------
    # TRAILS
    # --------------------------------------------------------

    def draw_trails(
        self,
        surface: pygame.Surface,
    ) -> None:

        for particle_index, particle in enumerate(
            self.system.particles
        ):

            trail = particle.trail

            if len(trail) < 2:
                continue

            step = max(
                1,
                len(trail) // 1000,
            )

            points = trail[::step]

            projected_points: list[
                tuple[ProjectedPoint, float]
            ] = []

            for index, point in enumerate(
                points
            ):

                projected = self.camera.project(
                    point,
                    SIMULATION_AREA,
                )

                if projected is None:
                    continue

                progress = (
                    index
                    / max(
                        1,
                        len(points) - 1,
                    )
                )

                projected_points.append(
                    (
                        projected,
                        progress,
                    )
                )

            if len(projected_points) < 2:
                continue

            for index in range(
                1,
                len(projected_points),
            ):

                previous, previous_progress = (
                    projected_points[index - 1]
                )

                current, current_progress = (
                    projected_points[index]
                )

                colour_amount = (
                    current_progress
                    * 0.85
                    + self.palette_shift
                )

                colour = palette_colour(
                    self.palette,
                    colour_amount,
                )

                fade = (
                    0.15
                    + current_progress * 0.85
                )

                colour = (
                    int(colour[0] * fade),
                    int(colour[1] * fade),
                    int(colour[2] * fade),
                )

                width = 1

                if current_progress > 0.85:
                    width = 2

                pygame.draw.line(
                    surface,
                    colour,
                    (
                        int(previous.x),
                        int(previous.y),
                    ),
                    (
                        int(current.x),
                        int(current.y),
                    ),
                    width,
                )

            # Current particle head
            current_point = projected_points[-1][0]

            radius = 3

            pygame.draw.circle(
                surface,
                WHITE,
                (
                    int(current_point.x),
                    int(current_point.y),
                ),
                radius,
            )

            if self.glow_enabled:

                glow_surface = pygame.Surface(
                    (40, 40),
                    pygame.SRCALPHA,
                )

                for radius_value in range(
                    18,
                    2,
                    -3,
                ):

                    alpha = int(
                        5
                        + 35
                        * (
                            1.0
                            - radius_value / 20.0
                        )
                    )

                    pygame.draw.circle(
                        glow_surface,
                        (
                            self.palette[1][0],
                            self.palette[1][1],
                            self.palette[1][2],
                            alpha,
                        ),
                        (20, 20),
                        radius_value,
                    )

                surface.blit(
                    glow_surface,
                    (
                        int(current_point.x - 20),
                        int(current_point.y - 20),
                    ),
                    special_flags=pygame.BLEND_ALPHA_SDL2,
                )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    def draw_sidebar(
        self,
        surface: pygame.Surface,
    ) -> None:

        rect = pygame.Rect(
            WIDTH - SIDEBAR_WIDTH - MARGIN,
            HEADER_HEIGHT + MARGIN,
            SIDEBAR_WIDTH,
            HEIGHT - HEADER_HEIGHT - MARGIN * 2,
        )

        self.ui.panel(
            surface,
            rect,
        )

        x = rect.x + 18

        y = rect.y + 18

        self.ui.text(
            surface,
            "CHAOS LAB",
            (x, y),
            "large",
            WHITE,
        )

        y += 40

        self.ui.text(
            surface,
            "LORENZ ATTRACTOR",
            (x, y),
            "small",
            CYAN,
        )

        y += 32

        self.ui.line(
            surface,
            (x, y),
            (rect.right - 18, y),
            GRID,
        )

        y += 20

        p = self.parameters

        lines = (
            ("SIGMA", f"{p.sigma:6.2f}"),
            ("RHO", f"{p.rho:6.2f}"),
            ("BETA", f"{p.beta:6.3f}"),
            ("SPEED", f"{self.system.speed:6.2f}x"),
            (
                "TRAIL",
                f"{self.system.trail_limit:6d}",
            ),
            (
                "PARTICLES",
                f"{len(self.system.particles):6d}",
            ),
        )

        for label, value in lines:

            self.ui.text(
                surface,
                label,
                (x, y),
                "small",
                MUTED,
            )

            self.ui.text(
                surface,
                value,
                (
                    rect.right - 18,
                    y,
                ),
                "small",
                WHITE,
                anchor="topright",
            )

            y += 24

        y += 6

        self.ui.line(
            surface,
            (x, y),
            (rect.right - 18, y),
            GRID,
        )

        y += 18

        self.ui.text(
            surface,
            "PRESETS",
            (x, y),
            "small",
            AMBER,
        )

        y += 25

        preset_lines = (
            "1  CLASSIC  10 / 28 / 2.667",
            "2  STABLE   10 / 20 / 2.667",
            "3  CHAOTIC  14 / 35 / 3.000",
        )

        for line in preset_lines:

            self.ui.text(
                surface,
                line,
                (x, y),
                "tiny",
                MUTED,
            )

            y += 20

        y += 8

        self.ui.text(
            surface,
            "CONTROLS",
            (x, y),
            "small",
            AMBER,
        )

        y += 24

        controls = (
            "SPACE   pause / resume",
            "R       reset",
            "J       randomize",
            "P       palette",
            "G       glow",
            "T       grid",
            "A       axes",
            "V       auto rotate",
            "M       audio",
            "S       screenshot",
            "C       clear trails",
            "+ / -   simulation speed",
            "[ / ]   trail length",
            "ARROWS  parameters",
            "Z / X   beta",
            "MOUSE   rotate / zoom",
        )

        for line in controls:

            self.ui.text(
                surface,
                line,
                (x, y),
                "tiny",
                MUTED,
            )

            y += 17

            if y > rect.bottom - 30:
                break

        status_y = rect.bottom - 24

        status = (
            "RUNNING"
            if self.system.running
            else "PAUSED"
        )

        status_colour = (
            GREEN
            if self.system.running
            else AMBER
        )

        self.ui.text(
            surface,
            status,
            (x, status_y),
            "small",
            status_colour,
            anchor="bottomleft",
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    def draw_header(
        self,
        surface: pygame.Surface,
    ) -> None:

        self.ui.text(
            surface,
            "EIDOLON",
            (MARGIN, 20),
            "title",
            WHITE,
        )

        self.ui.text(
            surface,
            "#011",
            (145, 23),
            "small",
            CYAN,
        )

        self.ui.text(
            surface,
            "CHAOS LAB",
            (200, 23),
            "small",
            MUTED,
        )

        self.ui.text(
            surface,
            f"PALETTE / {self.palette_name}",
            (
                WIDTH - MARGIN,
                22,
            ),
            "small",
            self.palette[1],
            anchor="topright",
        )

        self.ui.line(
            surface,
            (MARGIN, HEADER_HEIGHT - 1),
            (
                WIDTH - MARGIN,
                HEADER_HEIGHT - 1,
            ),
            GRID,
        )

    # --------------------------------------------------------
    # HUD
    # --------------------------------------------------------

    def draw_hud(
        self,
        surface: pygame.Surface,
    ) -> None:

        if not self.show_hud:
            return

        rect = pygame.Rect(
            SIMULATION_AREA.x + 14,
            SIMULATION_AREA.y + 14,
            245,
            90,
        )

        pygame.draw.rect(
            surface,
            (
                5,
                8,
                11,
            ),
            rect,
            border_radius=6,
        )

        pygame.draw.rect(
            surface,
            GRID,
            rect,
            1,
            border_radius=6,
        )

        p = self.parameters

        self.ui.text(
            surface,
            "LORENZ DYNAMICAL SYSTEM",
            (
                rect.x + 12,
                rect.y + 10,
            ),
            "tiny",
            CYAN,
        )

        self.ui.text(
            surface,
            (
                f"dx = {p.sigma:.1f}(y - x)"
            ),
            (
                rect.x + 12,
                rect.y + 30,
            ),
            "tiny",
            MUTED,
        )

        self.ui.text(
            surface,
            (
                f"dy = x({p.rho:.1f} - z) - y"
            ),
            (
                rect.x + 12,
                rect.y + 48,
            ),
            "tiny",
            MUTED,
        )

        self.ui.text(
            surface,
            (
                f"dz = xy - {p.beta:.3f}z"
            ),
            (
                rect.x + 12,
                rect.y + 66,
            ),
            "tiny",
            MUTED,
        )

    # --------------------------------------------------------
    # INTRO
    # --------------------------------------------------------

    def draw_intro(
        self,
        surface: pygame.Surface,
    ) -> None:

        if self.intro_timer <= 0:
            return

        alpha = int(
            clamp(
                self.intro_timer
                / INTRO_DURATION,
                0.0,
                1.0,
            )
            * 180
        )

        overlay = pygame.Surface(
            (WIDTH, HEIGHT),
            pygame.SRCALPHA,
        )

        overlay.fill(
            (
                0,
                0,
                0,
                alpha,
            )
        )

        surface.blit(
            overlay,
            (0, 0),
        )

        centre_x = WIDTH // 2
        centre_y = HEIGHT // 2

        self.ui.text(
            surface,
            "CHAOS",
            (
                centre_x,
                centre_y - 45,
            ),
            "title",
            WHITE,
            anchor="midbottom",
        )

        self.ui.text(
            surface,
            "IS NOT RANDOM",
            (
                centre_x,
                centre_y + 5,
            ),
            "large",
            CYAN,
            anchor="midtop",
        )

        self.ui.text(
            surface,
            "A tiny change can rewrite the trajectory.",
            (
                centre_x,
                centre_y + 42,
            ),
            "small",
            MUTED,
            anchor="midtop",
        )

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        dt: float,
    ) -> None:

        self.intro_timer -= dt

        if (
            self.intro_timer <= 0
            and not self.started
        ):

            self.started = True

        if self.camera.camera.auto_rotate:

            self.camera.camera.yaw += (
                dt * 0.35
            )

        self.palette_shift += dt * 0.04

        self.system.update(
            max(
                1,
                int(
                    3
                    * self.system.speed
                ),
            )
        )

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    def render(self) -> None:

        self.screen.fill(
            BLACK
        )

        self.draw_grid(
            self.screen
        )

        self.draw_axes(
            self.screen
        )

        self.draw_trails(
            self.screen
        )

        self.draw_header(
            self.screen
        )

        self.draw_sidebar(
            self.screen
        )

        self.draw_hud(
            self.screen
        )

        self.draw_intro(
            self.screen
        )

        pygame.display.flip()

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def run(self) -> None:

        while self.running:

            frame_start = time.perf_counter()

            for event in pygame.event.get():

                self.handle_event(
                    event
                )

            dt = self.clock.tick(
                FPS
            ) / 1000.0

            self.update(
                dt
            )

            self.render()

            self.frame_time = (
                time.perf_counter()
                - frame_start
            )

            instantaneous_fps = (
                1.0 / self.frame_time
                if self.frame_time > 0
                else 0.0
            )

            self.fps = lerp(
                self.fps,
                instantaneous_fps,
                0.08,
            )

        pygame.quit()


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "EIDOLON #011 — Chaos Lab"
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a mathematical simulation test.",
    )

    args = parser.parse_args()

    app = ChaosLab(
        smoke_test=args.smoke_test
    )

    app.run()


if __name__ == "__main__":
    main()