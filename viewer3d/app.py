"""Application: render loop (update(dt) / render()), input, resize/DPR, headless mode."""
import math
import os
import time

import moderngl
from PIL import Image

from .renderer import Renderer
from .scene import OrbitControls, PerspectiveCamera
from .scenes import frame_scene


class Viewer:
    """Owns the scene, camera, controls and renderer; independent of the window system."""

    def __init__(self, ctx, scene, samples=4):
        self.ctx = ctx
        self.scene = scene
        self.camera = PerspectiveCamera(fov=50.0)
        self.controls = OrbitControls(self.camera)
        self.controls.auto_rotate = 0.15
        frame_scene(scene, self.controls)
        self.renderer = Renderer(ctx, samples=samples)
        self.time = 0.0
        self.paused = False

    def set_viewport(self, fb_width, fb_height):
        self.camera.aspect = max(1, fb_width) / max(1, fb_height)

    def update(self, dt):
        if not self.paused:
            self.time += dt
        self.controls.update(dt)
        self.scene.update(dt, self.time)

    def render(self, target_fbo):
        self.renderer.render(self.scene, self.camera, target_fbo)

    def snapshot(self, size):
        """Render the current frame offscreen and return it as a PIL image.

        Reading the window's back buffer after a swap is undefined, so screenshots
        always go through their own framebuffer.
        """
        fbo = self.ctx.simple_framebuffer(size, components=4)
        self.render(fbo)
        img = read_fbo(fbo)
        fbo.release()
        return img


def read_fbo(fbo):
    w, h = fbo.size
    img = Image.frombytes("RGB", (w, h), fbo.read(components=3))
    return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def run_headless(scene, out_path, size=(1280, 720), frames=1, dt=1 / 60, camera=None, samples=4):
    """Render offscreen (EGL, no display needed) and save a PNG of the last frame."""
    try:
        ctx = moderngl.create_standalone_context(backend="egl")
    except Exception:
        ctx = moderngl.create_standalone_context()
    viewer = Viewer(ctx, scene, samples=samples)
    if camera:
        c = viewer.controls
        c.yaw = camera.get("yaw", c.yaw)
        c.pitch = camera.get("pitch", c.pitch)
        c.zoom(camera.get("zoom", 1.0))
        c.auto_rotate = camera.get("auto_rotate", c.auto_rotate)
    fbo = ctx.simple_framebuffer(size, components=4)
    viewer.set_viewport(*size)
    for _ in range(max(1, frames)):
        viewer.update(dt)
        viewer.render(fbo)
    img = read_fbo(fbo)
    if out_path:
        img.save(out_path)
    info = {"renderer": ctx.info["GL_RENDERER"], "gl": ctx.version_code,
            "samples": viewer.renderer.samples, **viewer.renderer.stats}
    ctx.release()
    return img, info


def run_window(scene, size=(1280, 720), title="3dseg viewer", samples=4, screenshot_dir=".",
               max_seconds=None, screenshot_on_exit=None):
    import glfw

    if not glfw.init():
        raise RuntimeError("glfw.init() failed (no display? use --headless)")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.SCALE_TO_MONITOR, True)  # honour monitor DPI scaling
    window = glfw.create_window(size[0], size[1], title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create an OpenGL 3.3 window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    try:
        ctx = moderngl.create_context()
    except OSError:
        # Distros without the libGL.so dev symlink only ship the versioned runtime library.
        ctx = moderngl.create_context(libgl="libGL.so.1")
    viewer = Viewer(ctx, scene, samples=samples)
    state = {"drag": None, "last": None, "idle": 0.0, "auto": True}

    def on_resize(_w, fb_w, fb_h):
        # Framebuffer size is in physical pixels (window size * DPR on HiDPI).
        viewer.set_viewport(fb_w, fb_h)

    def on_button(_w, button, action, mods):
        if action == glfw.PRESS:
            pan = button in (glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE) or mods & glfw.MOD_SHIFT
            state["drag"] = "pan" if pan else "orbit"
            state["last"] = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            state["drag"] = None

    def on_cursor(_w, x, y):
        if not state["drag"]:
            return
        lx, ly = state["last"]
        state["last"] = (x, y)
        # cursor coords are in window units; normalise by window height so
        # sensitivity is independent of DPR and window size
        h = max(1, glfw.get_window_size(window)[1])
        dx, dy = (x - lx) / h, (y - ly) / h
        if state["drag"] == "orbit":
            viewer.controls.rotate(-dx * math.pi, dy * math.pi)
        else:
            viewer.controls.pan(dx, dy)
        state["idle"] = 0.0

    def on_scroll(_w, _dx, dy):
        viewer.controls.zoom(0.9 ** dy)
        state["idle"] = 0.0

    def on_key(_w, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            viewer.paused = not viewer.paused
        elif key == glfw.KEY_R:
            frame_scene(viewer.scene, viewer.controls)
        elif key == glfw.KEY_A:
            state["auto"] = not state["auto"]
        elif key == glfw.KEY_S:
            path = os.path.join(screenshot_dir, f"screenshot_{int(time.time())}.png")
            viewer.snapshot(glfw.get_framebuffer_size(window)).save(path)
            print("saved", path)

    glfw.set_framebuffer_size_callback(window, on_resize)
    glfw.set_mouse_button_callback(window, on_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)
    on_resize(window, *glfw.get_framebuffer_size(window))

    prev = start = time.perf_counter()
    fps_t, fps_n = prev, 0
    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = min(now - prev, 0.1)  # clamp so a stall doesn't teleport animations
        prev = now
        # auto-orbit only after 3 s without user input
        state["idle"] += dt
        idle = state["idle"] > 3.0 and not state["drag"]
        viewer.controls.auto_rotate = 0.15 if state["auto"] and idle else 0.0

        viewer.update(dt)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        if fb_w > 0 and fb_h > 0:  # minimised windows have a 0x0 framebuffer
            screen = ctx.detect_framebuffer()
            viewer.render(screen)
            glfw.swap_buffers(window)
        glfw.poll_events()
        if max_seconds is not None and now - start >= max_seconds:
            glfw.set_window_should_close(window, True)

        fps_n += 1
        if now - fps_t >= 1.0:
            glfw.set_window_title(window, f"{title} — {fps_n / (now - fps_t):.0f} fps, "
                                          f"{viewer.renderer.stats.get('triangles', 0)} tris")
            fps_t, fps_n = now, 0
    if screenshot_on_exit:
        viewer.snapshot(glfw.get_framebuffer_size(window)).save(screenshot_on_exit)
    glfw.terminate()
