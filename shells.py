"""
Spectral Shell Visualizer - Constant-Q Standing Wave Synthesis

Standing wave visualizer with mirror symmetry about x and y axes.
Uses even-even cosine modes: cos(kx*x) * cos(ky*y)

Shell definition (constant-Q, q = 2^(1/4)):
    s(m,n) = floor(4 * log2(sqrt(m² + n²)))

Field synthesis (x and y symmetric):
    u(x,y,t) = Σ A_s(m,n)(t) / N_s * cos(kx*x) * cos(ky*y)

Energy rendering:
    I(x,y,t) = u(x,y,t)²
"""

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import soundcard as sc
import threading
import time
import ctypes
import warnings

# Suppress soundcard buffer underrun warnings
warnings.filterwarnings("ignore", message="data discontinuity in recording")

SAMPLE_RATE = 44100
CHUNK_SIZE = 64
MAX_FFT_SIZE = 32768
MAX_BINS = MAX_FFT_SIZE // 2 + 1

# Maximum number of modes to precompute
MAX_MODES = 2048
# Maximum shell index
MAX_SHELLS = 64

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
out vec2 fragCoord;
uniform vec2 iResolution;
void main() {
    fragCoord = (position + 1.0) * 0.5 * iResolution;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 fragCoord;
out vec4 outColor;

uniform vec2 iResolution;
uniform sampler1D iShellAmps;       // Shell amplitudes A_s(t)
uniform sampler1D iModeData;        // Packed mode data: (kx, ky, shell, inv_count)
uniform int iNumModes;
uniform int iNumShells;
uniform float iTime;
uniform int iColorMode;             // 0=Plasma, 1=Magma, 2=Turbo, 3=Viridis, 4=Grayscale
uniform float iContrast;
uniform float iScale;               // Spatial frequency scale
uniform int iModeType;              // 0=all, 1=m!=n only, 2=diagonal pairs

#define PI 3.14159265
#define TAU 6.28318530

// ============================================================================
// COLORMAPS
// ============================================================================

vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
    vec3 c1 = vec3(0.417642, 0.000564, 0.658390);
    vec3 c2 = vec3(0.692840, 0.165141, 0.564522);
    vec3 c3 = vec3(0.881443, 0.392529, 0.383229);
    vec3 c4 = vec3(0.987622, 0.645320, 0.039886);
    vec3 c5 = vec3(0.940015, 0.975158, 0.131326);
    if (t >= 1.0) return c5;
    float s = t * 5.0;
    int idx = int(floor(s));
    float f = fract(s);
    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    return mix(c4, c5, f);
}

vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001462, 0.000466, 0.013866);
    vec3 c1 = vec3(0.316654, 0.071862, 0.485380);
    vec3 c2 = vec3(0.716387, 0.214982, 0.474720);
    vec3 c3 = vec3(0.974417, 0.462840, 0.359756);
    vec3 c4 = vec3(0.995131, 0.766837, 0.534094);
    vec3 c5 = vec3(0.987053, 0.991438, 0.749504);
    if (t >= 1.0) return c5;
    float s = t * 5.0;
    int idx = int(floor(s));
    float f = fract(s);
    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    return mix(c4, c5, f);
}

vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.18995, 0.07176, 0.23217);
    vec3 c1 = vec3(0.25107, 0.25237, 0.63374);
    vec3 c2 = vec3(0.15992, 0.53830, 0.72889);
    vec3 c3 = vec3(0.09140, 0.74430, 0.54318);
    vec3 c4 = vec3(0.52876, 0.85393, 0.21546);
    vec3 c5 = vec3(0.88092, 0.73551, 0.07741);
    vec3 c6 = vec3(0.97131, 0.45935, 0.05765);
    vec3 c7 = vec3(0.84299, 0.15070, 0.15090);
    if (t >= 1.0) return c7;
    float s = t * 7.0;
    int idx = int(floor(s));
    float f = fract(s);
    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    if (idx == 4) return mix(c4, c5, f);
    if (idx == 5) return mix(c5, c6, f);
    return mix(c6, c7, f);
}

vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
    vec3 c1 = vec3(0.282327, 0.140926, 0.457517);
    vec3 c2 = vec3(0.253935, 0.265254, 0.529983);
    vec3 c3 = vec3(0.206756, 0.371758, 0.553117);
    vec3 c4 = vec3(0.143936, 0.522773, 0.556295);
    vec3 c5 = vec3(0.119512, 0.607464, 0.540218);
    vec3 c6 = vec3(0.166383, 0.690856, 0.496502);
    vec3 c7 = vec3(0.319809, 0.770914, 0.411152);
    vec3 c8 = vec3(0.525776, 0.833491, 0.288127);
    vec3 c9 = vec3(0.762373, 0.876424, 0.137064);
    vec3 c10 = vec3(0.993248, 0.906157, 0.143936);
    if (t >= 1.0) return c10;
    float s = t * 10.0;
    int idx = int(floor(s));
    float f = fract(s);
    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    if (idx == 4) return mix(c4, c5, f);
    if (idx == 5) return mix(c5, c6, f);
    if (idx == 6) return mix(c6, c7, f);
    if (idx == 7) return mix(c7, c8, f);
    if (idx == 8) return mix(c8, c9, f);
    return mix(c9, c10, f);
}

// ============================================================================
// MAIN
// ============================================================================

void main() {
    vec2 uv = fragCoord / iResolution;
    float aspect = iResolution.x / iResolution.y;

    // Normalized coordinates centered at origin, scaled
    vec2 p;
    p.x = (uv.x - 0.5) * 2.0 * aspect * iScale;
    p.y = (uv.y - 0.5) * 2.0 * iScale;

    // Field synthesis: u(x,y,t) = Σ A_s / N_s * cos(kx*x) * cos(ky*y)
    // Even-even symmetry: invariant under x → -x and y → -y
    float u = 0.0;

    for (int i = 0; i < 2048; i++) {
        if (i >= iNumModes) break;

        // Fetch mode data: (kx, ky, shell, inv_count)
        float texCoord = (float(i) + 0.5) / float(iNumModes);
        vec4 mode = texture(iModeData, texCoord);

        float kx = mode.r;
        float ky = mode.g;
        int shell = int(mode.b);
        float inv_count = mode.a;

        // Get shell amplitude
        float shellCoord = (float(shell) + 0.5) / float(iNumShells);
        float A_s = texture(iShellAmps, shellCoord).r;

        // Standing wave with x and y mirror symmetry
        float mode_val;

        if (iModeType == 0) {
            // All modes: cos(kx*x) * cos(ky*y)
            mode_val = cos(kx * p.x) * cos(ky * p.y);
        } else if (iModeType == 1) {
            // Only m != n modes (skip when kx == ky)
            if (abs(kx - ky) < 0.01) continue;
            mode_val = cos(kx * p.x) * cos(ky * p.y);
        } else {
            // Diagonal symmetry: cos(kx*x)*cos(ky*y) + cos(ky*x)*cos(kx*y)
            // This adds 4-fold rotational symmetry (90° invariance)
            mode_val = cos(kx * p.x) * cos(ky * p.y) + cos(ky * p.x) * cos(kx * p.y);
            if (abs(kx - ky) < 0.01) mode_val *= 0.5;  // Avoid double-counting m=n
        }

        u += A_s * inv_count * mode_val;
    }

    // Energy rendering: I = u²
    float I = u * u;

    // Apply contrast
    I = pow(I, 1.0 / iContrast);

    // Soft clamp
    I = tanh(I * 2.0);

    // Color mapping
    vec3 color;
    if (iColorMode == 0) {
        color = plasma(I);
    } else if (iColorMode == 1) {
        color = magma(I);
    } else if (iColorMode == 2) {
        color = turbo(I);
    } else if (iColorMode == 3) {
        color = viridis(I);
    } else {
        // Grayscale
        color = vec3(I);
    }

    outColor = vec4(color, 1.0);
}
"""


class AudioCapture:
    def __init__(self, fft_size=8192):
        self.running = False
        self.thread = None
        self.fft_size = fft_size
        self._ring_buffer = np.zeros(MAX_FFT_SIZE * 4, dtype=np.float32)
        self._write_pos = 0
        self._running_max = 0.01
        self._output = np.zeros(MAX_BINS, dtype=np.float32)
        self._update_fft_params()

    def _update_fft_params(self):
        self._n_bins = self.fft_size // 2 + 1
        self._window = np.hanning(self.fft_size).astype(np.float32)
        self._spectrum = np.zeros(self._n_bins, dtype=np.float32)
        self._running_max = 0.01

    def set_fft_size(self, size, viz=None):
        size = max(256, min(MAX_FFT_SIZE, size))
        if size != self.fft_size:
            self.fft_size = size
            self._update_fft_params()
            if viz:
                viz.print_status()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def _capture_loop(self):
        try:
            try:
                ctypes.windll.kernel32.SetThreadPriority(
                    ctypes.windll.kernel32.GetCurrentThread(), 2)
            except:
                pass

            speaker = sc.default_speaker()
            with sc.get_microphone(id=str(speaker.name), include_loopback=True).recorder(
                samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE) as mic:
                while self.running:
                    data = mic.record(numframes=CHUNK_SIZE)
                    if data.ndim > 1:
                        data = data.mean(axis=1)

                    chunk_len = len(data)
                    buf_len = len(self._ring_buffer)
                    end_pos = self._write_pos + chunk_len

                    if end_pos <= buf_len:
                        self._ring_buffer[self._write_pos:end_pos] = data
                    else:
                        first_part = buf_len - self._write_pos
                        self._ring_buffer[self._write_pos:] = data[:first_part]
                        self._ring_buffer[:chunk_len - first_part] = data[first_part:]

                    self._write_pos = end_pos % buf_len
        except Exception as e:
            print(f"Audio error: {e}")

    def get_spectrum(self):
        fft_size = self.fft_size
        n_bins = fft_size // 2 + 1

        current_pos = self._write_pos
        buf_len = len(self._ring_buffer)
        start = (current_pos - fft_size) % buf_len

        if start + fft_size <= buf_len:
            samples = self._ring_buffer[start:start + fft_size].copy()
        else:
            first_part = buf_len - start
            samples = np.concatenate([self._ring_buffer[start:], self._ring_buffer[:fft_size - first_part]])

        magnitude = np.abs(np.fft.rfft(samples * self._window[:fft_size]))
        self._spectrum[:n_bins] = magnitude

        current_max = np.max(self._spectrum[:n_bins])
        if current_max > self._running_max:
            self._running_max = current_max
        else:
            self._running_max = max(self._running_max * 0.995, current_max, 0.01)

        normalized = np.clip(self._spectrum[:n_bins] / (self._running_max + 1e-6), 0.0, 1.5)

        self._output[:] = 0
        self._output[:n_bins] = normalized
        return self._output, n_bins


class ShellProcessor:
    """Maps FFT bins to constant-Q shells and computes shell amplitudes."""

    def __init__(self, num_shells=MAX_SHELLS, kappa=5.0, tau_attack=0.040, tau_release=0.300):
        self.num_shells = num_shells
        self.kappa = kappa  # Log compression factor
        self.tau_attack = tau_attack
        self.tau_release = tau_release

        # Shell amplitudes A_s(t)
        self.shell_amps = np.zeros(num_shells, dtype=np.float32)

        # Precompute mode data
        self._init_modes()

        self._last_time = time.time()

    def _init_modes(self):
        """Precompute (m,n) modes and their shells. No random phases needed."""
        modes = []
        shell_counts = np.zeros(self.num_shells, dtype=np.int32)

        # Generate modes for m,n >= 0 only (symmetry handled by cos*cos)
        # cos(kx*x)*cos(ky*y) is already symmetric about x and y axes
        max_mn = 32  # Covers shells up to ~20

        for m in range(0, max_mn + 1):
            for n in range(0, max_mn + 1):
                if m == 0 and n == 0:  # Skip DC
                    continue

                r = np.sqrt(m*m + n*n)

                # Shell index: s = floor(4 * log2(r))
                s = int(np.floor(4 * np.log2(r)))
                if s < 0 or s >= self.num_shells:
                    continue

                # Wave vector (using π scaling for nice periodicity)
                kx = m * np.pi
                ky = n * np.pi

                modes.append((kx, ky, s))
                shell_counts[s] += 1

        self.modes = modes
        self.shell_counts = shell_counts
        self.num_modes = len(modes)

        # Pack mode data for GPU: (kx, ky, shell, 1/N_s)
        self.mode_data = np.zeros((self.num_modes, 4), dtype=np.float32)
        for i, (kx, ky, s) in enumerate(modes):
            inv_count = 1.0 / max(1, shell_counts[s])
            self.mode_data[i] = [kx, ky, float(s), inv_count]

        print(f"Initialized {self.num_modes} modes across {np.sum(shell_counts > 0)} active shells")

    def _bin_to_shell(self, bin_idx, n_bins, freq_per_bin):
        """Map FFT bin to shell index using constant-Q."""
        freq = bin_idx * freq_per_bin
        if freq < 20:  # Below audible
            return -1

        # Map frequency to approximate (m,n) radius
        # Assume base frequency maps to r=1
        base_freq = 40.0  # Hz
        r = freq / base_freq

        if r < 1:
            return -1

        # Shell index: s = floor(4 * log2(r))
        s = int(np.floor(4 * np.log2(r)))
        return min(s, self.num_shells - 1)

    def process(self, spectrum, n_bins, fft_size):
        """Process spectrum into shell amplitudes with attack/release dynamics."""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        freq_per_bin = SAMPLE_RATE / fft_size

        # Compute raw shell energies from FFT
        shell_energy_raw = np.zeros(self.num_shells, dtype=np.float32)

        for i in range(1, n_bins):
            s = self._bin_to_shell(i, n_bins, freq_per_bin)
            if 0 <= s < self.num_shells:
                shell_energy_raw[s] += spectrum[i] ** 2

        # Log compression: E_s = log(1 + κ * E_raw)
        E_s = np.log(1 + self.kappa * shell_energy_raw)

        # Attack/release smoothing
        # α = 1 - exp(-dt/τ)
        alpha_attack = 1 - np.exp(-dt / self.tau_attack)
        alpha_release = 1 - np.exp(-dt / self.tau_release)

        # A_s ← A_s + α * (E_s - A_s)
        rising = E_s > self.shell_amps
        self.shell_amps[rising] += alpha_attack * (E_s[rising] - self.shell_amps[rising])
        self.shell_amps[~rising] += alpha_release * (E_s[~rising] - self.shell_amps[~rising])

        return self.shell_amps


class Visualizer:
    DEFAULT_CONTRAST = 1.5
    DEFAULT_SCALE = 1.0
    DEFAULT_COLOR_MODE = 0  # Plasma
    DEFAULT_FFT_SIZE = 8192
    DEFAULT_KAPPA = 5.0
    DEFAULT_TAU_ATTACK = 0.040
    DEFAULT_TAU_RELEASE = 0.300

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._fft_size = self.DEFAULT_FFT_SIZE
        self._color_names = ["Plasma", "Magma", "Turbo", "Viridis", "Grayscale"]
        self._mode_type_names = ["All", "m!=n", "Diagonal"]
        self._time = 0.0
        self.reset_defaults()

    def reset_defaults(self):
        self.contrast = self.DEFAULT_CONTRAST
        self.scale = self.DEFAULT_SCALE
        self.color_mode = self.DEFAULT_COLOR_MODE
        self.mode_type = 0  # All modes
        self.kappa = self.DEFAULT_KAPPA
        self.tau_attack = self.DEFAULT_TAU_ATTACK
        self.tau_release = self.DEFAULT_TAU_RELEASE
        self.print_status()

    def init(self, shell_processor):
        self.shell_processor = shell_processor

        self.program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
            validate=False
        )

        verts = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
        inds = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        vbo, ebo = glGenBuffers(2)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.nbytes, inds, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(0)

        # 1D texture for shell amplitudes
        self.shell_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.shell_tex)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, MAX_SHELLS, 0, GL_RED, GL_FLOAT, None)

        # 1D texture for mode data (RGBA: kx, ky, theta, shell+inv_count)
        self.mode_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.mode_tex)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, shell_processor.num_modes, 0,
                     GL_RGBA, GL_FLOAT, shell_processor.mode_data)

        self.locs = {
            'iResolution': glGetUniformLocation(self.program, "iResolution"),
            'iShellAmps': glGetUniformLocation(self.program, "iShellAmps"),
            'iModeData': glGetUniformLocation(self.program, "iModeData"),
            'iNumModes': glGetUniformLocation(self.program, "iNumModes"),
            'iNumShells': glGetUniformLocation(self.program, "iNumShells"),
            'iTime': glGetUniformLocation(self.program, "iTime"),
            'iColorMode': glGetUniformLocation(self.program, "iColorMode"),
            'iContrast': glGetUniformLocation(self.program, "iContrast"),
            'iScale': glGetUniformLocation(self.program, "iScale"),
            'iModeType': glGetUniformLocation(self.program, "iModeType"),
        }

    def set_contrast(self, value):
        self.contrast = max(0.1, min(5.0, value))
        self.print_status()

    def set_scale(self, value):
        self.scale = max(0.1, min(10.0, value))
        self.print_status()

    def set_kappa(self, value):
        self.kappa = max(0.1, min(50.0, value))
        self.shell_processor.kappa = self.kappa
        self.print_status()

    def set_tau_attack(self, value):
        self.tau_attack = max(0.001, min(0.5, value))
        self.shell_processor.tau_attack = self.tau_attack
        self.print_status()

    def set_tau_release(self, value):
        self.tau_release = max(0.01, min(2.0, value))
        self.shell_processor.tau_release = self.tau_release
        self.print_status()

    def print_status(self):
        color_str = self._color_names[self.color_mode]
        mode_str = self._mode_type_names[self.mode_type]
        status = (f"Scale={self.scale:.2f}  Contrast={self.contrast:.1f}  "
                  f"Kappa={self.kappa:.1f}  Attack={self.tau_attack*1000:.0f}ms  "
                  f"Release={self.tau_release*1000:.0f}ms  "
                  f"Mode={mode_str}  Color={color_str}  FFT={self._fft_size}")
        print(status)

    def render(self, shell_amps, fft_size, dt):
        self._fft_size = fft_size
        self._time += dt

        # Update shell amplitudes texture
        glBindTexture(GL_TEXTURE_1D, self.shell_tex)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, len(shell_amps), GL_RED, GL_FLOAT, shell_amps)

        glViewport(0, 0, self.w, self.h)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glUniform2f(self.locs['iResolution'], self.w, self.h)
        glUniform1i(self.locs['iNumModes'], self.shell_processor.num_modes)
        glUniform1i(self.locs['iNumShells'], MAX_SHELLS)
        glUniform1f(self.locs['iTime'], self._time)
        glUniform1i(self.locs['iColorMode'], self.color_mode)
        glUniform1f(self.locs['iContrast'], self.contrast)
        glUniform1f(self.locs['iScale'], self.scale)
        glUniform1i(self.locs['iModeType'], self.mode_type)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_1D, self.shell_tex)
        glUniform1i(self.locs['iShellAmps'], 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_1D, self.mode_tex)
        glUniform1i(self.locs['iModeData'], 1)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


def main():
    if not glfw.init():
        return

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    screen_w, screen_h = mode.size.width, mode.size.height

    win_w, win_h = 1280, 720
    glfw.window_hint(glfw.DECORATED, glfw.TRUE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(win_w, win_h, "Spectral Shell Visualizer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_window_pos(window, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    shell_proc = ShellProcessor()
    viz = Visualizer(win_w, win_h)
    viz.init(shell_proc)
    audio = AudioCapture()
    audio.start()
    time.sleep(0.05)

    def on_resize(window, width, height):
        if width > 0 and height > 0:
            viz.w, viz.h = width, height
            glViewport(0, 0, width, height)

    glfw.set_framebuffer_size_callback(window, on_resize)

    is_fullscreen = False
    windowed_pos = glfw.get_window_pos(window)
    windowed_size = (win_w, win_h)

    def toggle_fullscreen():
        nonlocal is_fullscreen, windowed_pos, windowed_size
        if is_fullscreen:
            glfw.set_window_monitor(window, None, windowed_pos[0], windowed_pos[1],
                                    windowed_size[0], windowed_size[1], 0)
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
            viz.w, viz.h = windowed_size
            is_fullscreen = False
        else:
            windowed_pos = glfw.get_window_pos(window)
            windowed_size = glfw.get_window_size(window)
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_monitor(window, None, 0, 0, screen_w, screen_h, 0)
            viz.w, viz.h = screen_w, screen_h
            is_fullscreen = True

    print("Spectral Shell Visualizer - Constant-Q Standing Wave Synthesis")
    print("Controls: W/S=scale  Z/X=contrast  K/L=kappa  UP/DOWN=attack  LEFT/RIGHT=release")
    print("          M=mode type  V=color  F=FFT size  ALT+ENTER=fullscreen  SPACE=reset  ESC=quit")
    print()
    viz.print_status()

    prev_keys = {}
    key_cooldowns = {}
    repeat_delay = 0.1
    last_time = time.time()

    def key_pressed(key):
        curr = glfw.get_key(window, key) == glfw.PRESS
        prev = prev_keys.get(key, False)
        prev_keys[key] = curr
        return curr and not prev

    def key_ready(key):
        now = time.time()
        if glfw.get_key(window, key) != glfw.PRESS:
            return False
        last = key_cooldowns.get(key, 0)
        if now - last >= repeat_delay:
            key_cooldowns[key] = now
            return True
        return False

    while not glfw.window_should_close(window):
        now = time.time()
        dt = now - last_time
        last_time = now

        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        alt_held = (glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS or
                    glfw.get_key(window, glfw.KEY_RIGHT_ALT) == glfw.PRESS)
        if alt_held and key_pressed(glfw.KEY_ENTER):
            toggle_fullscreen()

        if key_ready(glfw.KEY_W):
            viz.set_scale(viz.scale * 1.1)
        if key_ready(glfw.KEY_S):
            viz.set_scale(viz.scale / 1.1)

        if key_ready(glfw.KEY_X):
            viz.set_contrast(viz.contrast + 0.1)
        if key_ready(glfw.KEY_Z):
            viz.set_contrast(viz.contrast - 0.1)

        if key_ready(glfw.KEY_L):
            viz.set_kappa(viz.kappa * 1.2)
        if key_ready(glfw.KEY_K):
            viz.set_kappa(viz.kappa / 1.2)

        if key_ready(glfw.KEY_UP):
            viz.set_tau_attack(viz.tau_attack * 1.2)
        if key_ready(glfw.KEY_DOWN):
            viz.set_tau_attack(viz.tau_attack / 1.2)

        if key_ready(glfw.KEY_RIGHT):
            viz.set_tau_release(viz.tau_release * 1.2)
        if key_ready(glfw.KEY_LEFT):
            viz.set_tau_release(viz.tau_release / 1.2)

        if key_pressed(glfw.KEY_F):
            new_size = audio.fft_size * 2 if audio.fft_size < MAX_FFT_SIZE else 256
            audio.set_fft_size(new_size, viz)

        if key_pressed(glfw.KEY_M):
            viz.mode_type = (viz.mode_type + 1) % len(viz._mode_type_names)
            viz.print_status()

        if key_pressed(glfw.KEY_V):
            viz.color_mode = (viz.color_mode + 1) % len(viz._color_names)
            viz.print_status()

        if key_pressed(glfw.KEY_SPACE):
            viz.reset_defaults()
            shell_proc.kappa = viz.kappa
            shell_proc.tau_attack = viz.tau_attack
            shell_proc.tau_release = viz.tau_release
            audio.set_fft_size(Visualizer.DEFAULT_FFT_SIZE, viz)

        spectrum, n_bins = audio.get_spectrum()
        shell_amps = shell_proc.process(spectrum, n_bins, audio.fft_size)
        viz.render(shell_amps, audio.fft_size, dt)
        glfw.swap_buffers(window)

    print()
    audio.stop()
    glfw.terminate()


if __name__ == "__main__":
    main()

