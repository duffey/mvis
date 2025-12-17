"""
CQT Shell Visualizer - Constant-Q Transform Standing Wave Synthesis

Standing wave visualizer with mirror symmetry about x and y axes.
Uses even-even cosine modes: cos(kx*x) * cos(ky*y)

Constant-Q Transform (CQT):
    - Frequency bins geometrically spaced (12 bins per octave = musical notes)
    - Variable window lengths: longer for bass, shorter for treble
    - Matches human perception of time-frequency tradeoff
    - Great for: chords, key detection, melody tracking, harmonic analysis

Field synthesis (x and y symmetric):
    u(x,y,t) = Σ A_s(m,n)(t) / N_s * cos(kx*x) * cos(ky*y)

Rendering:
    Water-like surface with specular highlights and fresnel rim lighting.
    Wave field u is treated as a height map; gradient gives surface normals.
    96 bins = 8 octaves × 12 semitones (C1 ~32Hz to B8 ~7902Hz)

Water surface lighting model adapted from "Seascape" by Alexander Alekseev (TDM), 2014
https://www.shadertoy.com/view/Ms2SD1
License: CC BY-NC-SA 3.0 (https://creativecommons.org/licenses/by-nc-sa/3.0/)
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

# CQT parameters
BINS_PER_OCTAVE = 12  # Musical: 12 semitones per octave
NUM_OCTAVES = 8       # C1 (~32 Hz) to B8 (~7902 Hz)
NUM_CQT_BINS = BINS_PER_OCTAVE * NUM_OCTAVES  # 96 bins total
F_MIN = 32.70         # C1 in Hz
Q_FACTOR = 1.0 / (2 ** (1 / BINS_PER_OCTAVE) - 1)  # Quality factor ~16.82

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
uniform sampler1D iCQTAmps;         // CQT bin amplitudes (96 bins = 8 octaves × 12 semitones)
uniform sampler1D iModeData;        // Packed mode data: (kx, ky, cqt_bin, inv_count)
uniform int iNumModes;
uniform int iNumCQTBins;
uniform float iTime;
uniform float iScale;               // Spatial frequency scale
uniform int iModeType;              // 0=all, 1=m!=n only, 2=diagonal pairs
uniform float iTotalEnergy;         // Total energy for normalization
uniform float iAmplitude;           // Amplitude scaling factor

#define PI 3.14159265

// ============================================================================
// DITHERING
// ============================================================================

// Hash function for pseudo-random noise
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Triangular dithering noise (-0.5 to 0.5, triangular distribution)
float triangularNoise(vec2 p) {
    float r1 = hash(p);
    float r2 = hash(p + vec2(1.0, 0.0));
    return (r1 + r2) * 0.5 - 0.5;
}

// ============================================================================
// WAVE FIELD COMPUTATION
// ============================================================================

// Rendering parameters
const float gradientScale = 2.0;        // Scale for gradient calculation

// Compute wave field value at position p (normalized by total energy)
float computeWaveField(vec2 p) {
    float u = 0.0;
    for (int i = 0; i < 2048; i++) {
        if (i >= iNumModes) break;

        float texCoord = (float(i) + 0.5) / float(iNumModes);
        vec4 mode = texture(iModeData, texCoord);

        float kx = mode.r;
        float ky = mode.g;
        int cqt_bin = int(mode.b);
        float inv_count = mode.a;

        float cqtCoord = (float(cqt_bin) + 0.5) / float(iNumCQTBins);
        float A_s = texture(iCQTAmps, cqtCoord).r;

        float mode_val;
        if (iModeType == 0) {
            mode_val = cos(kx * p.x) * cos(ky * p.y);
        } else if (iModeType == 1) {
            if (abs(kx - ky) < 0.01) continue;
            mode_val = cos(kx * p.x) * cos(ky * p.y);
        } else {
            mode_val = cos(kx * p.x) * cos(ky * p.y) + cos(ky * p.x) * cos(kx * p.y);
            if (abs(kx - ky) < 0.01) mode_val *= 0.5;
        }

        u += A_s * inv_count * mode_val;
    }
    // Normalize by total energy
    return u * iAmplitude / max(sqrt(iTotalEnergy), 0.001);
}

// ============================================================================
// LIGHTING
// ============================================================================

// Night sky background
vec3 getSkyColor(vec3 rd) {
    float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd) * 0.5 + 0.5;
    sd = pow(sd, 5.0);
    vec3 col = mix(vec3(0.05, 0.1, 0.2), vec3(0.1, 0.05, 0.2), sd);
    return col * 0.63;
}

// Soft diffuse lighting
float diffuse(vec3 n, vec3 l, float p) {
    return pow(dot(n, l) * 0.4 + 0.6, p);
}

// Normalized specular (energy conserving)
float specularHighlight(vec3 n, vec3 l, vec3 e, float s) {
    float nrm = (s + 8.0) / (PI * 8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

// Surface color with height-based variation
vec3 getSurfaceColor(vec3 n, vec3 light, vec3 eye, float dist, float height) {
    // Fresnel: edges reflect more (cubic falloff, clamped)
    float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fresnel = min(pow(fresnel, 3.0), 0.5);

    // Reflected sky
    vec3 reflected = getSkyColor(reflect(eye, n));

    // Refracted/subsurface color - darker water for specular contrast
    vec3 baseColor = vec3(0.005, 0.01, 0.025);
    vec3 surfaceColor = vec3(0.05, 0.1, 0.15);
    vec3 refracted = baseColor + diffuse(n, light, 80.0) * surfaceColor * 0.05;

    // Blend refracted and reflected based on fresnel
    vec3 color = mix(refracted, reflected, fresnel);

    // Distance attenuation
    float atten = max(1.0 - dist * dist * 0.001, 0.0);
    color += surfaceColor * atten * 0.02;

    // Height-based color - wave peaks glow brighter
    color += surfaceColor * height * 0.5 * atten;

    // Specular with distance-dependent power (inversesqrt like Seascape)
    float specPower = 600.0 * inversesqrt(max(dist * dist, 0.01));
    color += vec3(0.8, 0.9, 1.0) * specularHighlight(n, light, eye, specPower) * 1.5;

    return color;
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

    // Small offset for gradient calculation (in normalized space)
    float eps = 0.01 * iScale;

    // Compute wave field at current position
    float u = computeWaveField(p);

    // Compute gradient using central differences
    float u_px = computeWaveField(p + vec2(eps, 0.0));
    float u_mx = computeWaveField(p - vec2(eps, 0.0));
    float u_py = computeWaveField(p + vec2(0.0, eps));
    float u_my = computeWaveField(p - vec2(0.0, eps));

    vec2 gradient = vec2(u_px - u_mx, u_py - u_my) / (2.0 * eps);
    gradient *= gradientScale;

    // Surface normal from gradient (wave acts as height field)
    // Tilt slightly toward viewer for better 3D effect
    vec3 normal = normalize(vec3(-gradient.x, -gradient.y, 1.0));

    // Simulate a 3D view: eye looking down at slight angle
    vec3 eye = normalize(vec3(uv.x - 0.5, uv.y - 0.5, -1.0));

    // Light from upper-right, slightly behind viewer
    vec3 light = normalize(vec3(0.0, 1.0, 0.8));

    // Distance from center (for attenuation effects)
    float dist = length(p);

    // Get surface color with full lighting model
    vec3 color = getSurfaceColor(normal, light, eye, dist, u);

    // Apply gamma correction (linear to sRGB)
    color = pow(color, vec3(0.65));

    // Apply dithering to reduce banding (±0.5/255 in each channel)
    float dither = triangularNoise(fragCoord) / 255.0;
    color += dither;

    outColor = vec4(color, 1.0);
}
"""


class AudioCapture:
    """Audio capture with Constant-Q Transform.

    CQT uses geometrically-spaced frequency bins (12 per octave) with
    variable window lengths - longer for bass, shorter for treble.
    This matches musical perception and note spacing.
    """

    def __init__(self):
        self.running = False
        self.thread = None

        # Large ring buffer for long bass windows
        self._ring_buffer = np.zeros(MAX_FFT_SIZE * 4, dtype=np.float32)
        self._write_pos = 0
        self._running_max = 0.01

        # Precompute CQT kernels
        self._init_cqt_kernels()

        # Output
        self._cqt_output = np.zeros(NUM_CQT_BINS, dtype=np.float32)

    def _init_cqt_kernels(self):
        """Precompute CQT analysis kernels for each frequency bin.

        For each bin k:
            f_k = f_min * 2^(k/bins_per_octave)
            N_k = ceil(Q * fs / f_k)  # Window length (longer for low freq)
            kernel_k = window(N_k) * exp(-2πi * Q * n / N_k)
        """
        self.cqt_freqs = np.zeros(NUM_CQT_BINS)
        self.cqt_window_lens = np.zeros(NUM_CQT_BINS, dtype=np.int32)
        self.cqt_kernels = []

        for k in range(NUM_CQT_BINS):
            # Center frequency for this bin
            freq = F_MIN * (2 ** (k / BINS_PER_OCTAVE))
            self.cqt_freqs[k] = freq

            # Window length: N_k = Q * fs / f_k
            # Longer windows for low frequencies, shorter for high
            N_k = int(np.ceil(Q_FACTOR * SAMPLE_RATE / freq))
            N_k = min(N_k, MAX_FFT_SIZE)  # Cap at max buffer
            self.cqt_window_lens[k] = N_k

            # Complex exponential kernel with Hann window
            n = np.arange(N_k)
            window = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_k)  # Hann window
            kernel = window * np.exp(-2j * np.pi * Q_FACTOR * n / N_k)
            kernel = kernel / N_k  # Normalize

            self.cqt_kernels.append(kernel.astype(np.complex64))

        max_window = np.max(self.cqt_window_lens)
        min_window = np.min(self.cqt_window_lens)
        print(f"CQT initialized: {NUM_CQT_BINS} bins, {BINS_PER_OCTAVE}/octave")
        print(f"  Freq range: {self.cqt_freqs[0]:.1f} Hz - {self.cqt_freqs[-1]:.1f} Hz")
        print(f"  Window range: {min_window} - {max_window} samples")
        print(f"  Time resolution: {min_window/SAMPLE_RATE*1000:.1f} - {max_window/SAMPLE_RATE*1000:.1f} ms")

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

    def get_cqt(self):
        """Compute Constant-Q Transform using precomputed kernels.

        Each bin uses a different window length, providing:
        - Better frequency resolution for bass (long windows)
        - Better time resolution for treble (short windows)
        """
        current_pos = self._write_pos
        buf_len = len(self._ring_buffer)

        for k in range(NUM_CQT_BINS):
            N_k = self.cqt_window_lens[k]
            kernel = self.cqt_kernels[k]

            # Extract samples for this bin's window length
            start = (current_pos - N_k) % buf_len

            if start + N_k <= buf_len:
                samples = self._ring_buffer[start:start + N_k]
            else:
                first_part = buf_len - start
                samples = np.concatenate([
                    self._ring_buffer[start:],
                    self._ring_buffer[:N_k - first_part]
                ])

            # Convolve with kernel (dot product for matched frequency)
            self._cqt_output[k] = np.abs(np.dot(samples, kernel))

        # Adaptive normalization with slower response and much higher floor
        # This prevents over-saturation at loud volumes
        current_max = np.max(self._cqt_output)

        # Slow attack (don't jump up instantly), slower release
        if current_max > self._running_max:
            # Very slow attack
            self._running_max += 0.01 * (current_max - self._running_max)
        else:
            # Very slow release
            self._running_max = max(self._running_max * 0.999, current_max, 0.0001)

        # Use much higher reference point to leave lots of headroom
        # 5x headroom means normal volume sits around 20% amplitude
        reference = self._running_max * 5.0 + 0.0001

        normalized = np.clip(self._cqt_output / reference, 0.0, 1.0)

        return normalized


class CQTProcessor:
    """Maps CQT bins to spatial modes for visualization.

    Each CQT bin (musical note) drives a set of spatial modes.
    96 bins = 8 octaves × 12 semitones (C1 to B8).
    """

    def __init__(self, kappa=5.0, tau_attack=0.040, tau_release=0.300):
        self.num_bins = NUM_CQT_BINS
        self.kappa = kappa  # Log compression factor
        self.tau_attack = tau_attack
        self.tau_release = tau_release

        # CQT bin amplitudes A_k(t)
        self.cqt_amps = np.zeros(NUM_CQT_BINS, dtype=np.float32)

        # Precompute mode data
        self._init_modes()

        self._last_time = time.time()

    def _init_modes(self):
        """Precompute simple modes - one per CQT bin.

        Much simpler mapping: each musical note gets ONE distinct spatial mode.
        Lower notes = lower spatial frequency, higher notes = higher spatial frequency.
        This creates clear, readable patterns tied directly to the music.
        """
        modes = []
        bin_counts = np.zeros(self.num_bins, dtype=np.int32)

        # Simple approach: each CQT bin gets one mode
        # Spatial frequency grows with pitch
        # Use different (m,n) combinations for variety within each octave

        # Mode patterns for the 12 semitones (gives visual variety)
        # These are (m,n) offsets that create different shapes
        semitone_patterns = [
            (1, 0),   # C  - horizontal
            (1, 1),   # C# - diagonal
            (0, 1),   # D  - vertical
            (2, 1),   # D# - angled
            (1, 2),   # E  - angled other way
            (2, 0),   # F  - horizontal 2nd harmonic
            (2, 2),   # F# - diagonal 2nd
            (0, 2),   # G  - vertical 2nd
            (3, 1),   # G# - complex
            (1, 3),   # A  - complex other
            (3, 2),   # A# - complex
            (2, 3),   # B  - complex other
        ]

        for k in range(self.num_bins):
            octave = k // BINS_PER_OCTAVE  # 0-7
            semitone = k % BINS_PER_OCTAVE  # 0-11

            # Base spatial frequency scales with octave (doubles each octave)
            base_freq = 2 ** (octave / 2.0)  # Slower growth for visibility

            # Get pattern for this semitone
            m_base, n_base = semitone_patterns[semitone]

            # Scale by octave
            m = max(1, int(m_base * base_freq))
            n = max(0, int(n_base * base_freq))

            # Ensure we don't have (0,0)
            if m == 0 and n == 0:
                m = 1

            kx = m * np.pi
            ky = n * np.pi

            modes.append((kx, ky, k))
            bin_counts[k] = 1

        self.modes = modes
        self.bin_counts = bin_counts
        self.num_modes = len(modes)

        # Pack mode data for GPU: (kx, ky, cqt_bin, 1/N_k)
        self.mode_data = np.zeros((self.num_modes, 4), dtype=np.float32)
        for i, (kx, ky, k) in enumerate(modes):
            inv_count = 1.0  # Each bin has exactly one mode
            self.mode_data[i] = [kx, ky, float(k), inv_count]

        print(f"Simple mode mapping: {self.num_modes} modes (1 per CQT bin)")

    def process(self, cqt_spectrum):
        """Process CQT spectrum into smoothed amplitudes with attack/release dynamics."""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # Log compression: E_k = log(1 + κ * spectrum²)
        E_k = np.log(1 + self.kappa * cqt_spectrum ** 2)

        # Attack/release smoothing
        alpha_attack = 1 - np.exp(-dt / self.tau_attack)
        alpha_release = 1 - np.exp(-dt / self.tau_release)

        # A_k ← A_k + α * (E_k - A_k)
        rising = E_k > self.cqt_amps
        self.cqt_amps[rising] += alpha_attack * (E_k[rising] - self.cqt_amps[rising])
        self.cqt_amps[~rising] += alpha_release * (E_k[~rising] - self.cqt_amps[~rising])

        return self.cqt_amps


class Visualizer:
    DEFAULT_SCALE = 0.5
    DEFAULT_KAPPA = 1.0  # Lower = more dynamic range, less compression
    DEFAULT_TAU_ATTACK = 0.040
    DEFAULT_TAU_RELEASE = 0.300
    DEFAULT_AMPLITUDE = 0.1

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._mode_type_names = ["All", "m!=n", "Diagonal"]
        self._time = 0.0
        self.reset_defaults()

    def reset_defaults(self):
        self.scale = self.DEFAULT_SCALE
        self.mode_type = 0  # All modes
        self.kappa = self.DEFAULT_KAPPA
        self.tau_attack = self.DEFAULT_TAU_ATTACK
        self.tau_release = self.DEFAULT_TAU_RELEASE
        self.amplitude = self.DEFAULT_AMPLITUDE
        self.print_status()

    def init(self, cqt_processor):
        self.cqt_processor = cqt_processor

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

        # 1D texture for CQT amplitudes (96 bins)
        self.cqt_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.cqt_tex)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NUM_CQT_BINS, 0, GL_RED, GL_FLOAT, None)

        # 1D texture for mode data (RGBA: kx, ky, cqt_bin, inv_count)
        self.mode_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.mode_tex)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, cqt_processor.num_modes, 0,
                     GL_RGBA, GL_FLOAT, cqt_processor.mode_data)

        self.locs = {
            'iResolution': glGetUniformLocation(self.program, "iResolution"),
            'iCQTAmps': glGetUniformLocation(self.program, "iCQTAmps"),
            'iModeData': glGetUniformLocation(self.program, "iModeData"),
            'iNumModes': glGetUniformLocation(self.program, "iNumModes"),
            'iNumCQTBins': glGetUniformLocation(self.program, "iNumCQTBins"),
            'iTime': glGetUniformLocation(self.program, "iTime"),
            'iScale': glGetUniformLocation(self.program, "iScale"),
            'iModeType': glGetUniformLocation(self.program, "iModeType"),
            'iTotalEnergy': glGetUniformLocation(self.program, "iTotalEnergy"),
            'iAmplitude': glGetUniformLocation(self.program, "iAmplitude"),
        }

    def set_scale(self, value):
        self.scale = max(0.1, min(10.0, value))
        self.print_status()

    def set_kappa(self, value):
        self.kappa = max(0.1, min(50.0, value))
        self.cqt_processor.kappa = self.kappa
        self.print_status()

    def set_tau_attack(self, value):
        self.tau_attack = max(0.001, min(0.5, value))
        self.cqt_processor.tau_attack = self.tau_attack
        self.print_status()

    def set_tau_release(self, value):
        self.tau_release = max(0.01, min(2.0, value))
        self.cqt_processor.tau_release = self.tau_release
        self.print_status()

    def set_amplitude(self, value):
        self.amplitude = max(0.01, min(5.0, value))
        self.print_status()

    def print_status(self):
        mode_str = self._mode_type_names[self.mode_type]
        status = (f"Scale={self.scale:.2f}  Amp={self.amplitude:.2f}  "
                  f"Kappa={self.kappa:.1f}  Attack={self.tau_attack*1000:.0f}ms  "
                  f"Release={self.tau_release*1000:.0f}ms  "
                  f"Mode={mode_str}")
        print(status)

    def render(self, cqt_amps, dt):
        self._time += dt

        # Compute total energy (sum of squared amplitudes)
        total_energy = np.sum(cqt_amps ** 2)

        # Update CQT amplitudes texture
        glBindTexture(GL_TEXTURE_1D, self.cqt_tex)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, len(cqt_amps), GL_RED, GL_FLOAT, cqt_amps)

        glViewport(0, 0, self.w, self.h)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glUniform2f(self.locs['iResolution'], self.w, self.h)
        glUniform1i(self.locs['iNumModes'], self.cqt_processor.num_modes)
        glUniform1i(self.locs['iNumCQTBins'], NUM_CQT_BINS)
        glUniform1f(self.locs['iTime'], self._time)
        glUniform1f(self.locs['iScale'], self.scale)
        glUniform1i(self.locs['iModeType'], self.mode_type)
        glUniform1f(self.locs['iTotalEnergy'], float(total_energy))
        glUniform1f(self.locs['iAmplitude'], self.amplitude)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_1D, self.cqt_tex)
        glUniform1i(self.locs['iCQTAmps'], 0)

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

    window = glfw.create_window(win_w, win_h, "CQT Shell Visualizer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_window_pos(window, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cqt_proc = CQTProcessor()
    viz = Visualizer(win_w, win_h)
    viz.init(cqt_proc)

    # Sync visualizer defaults to processor
    cqt_proc.kappa = viz.kappa
    cqt_proc.tau_attack = viz.tau_attack
    cqt_proc.tau_release = viz.tau_release

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

    print("CQT Shell Visualizer - Constant-Q Transform Standing Wave Synthesis")
    print("Controls: W/S=scale  Z/X=amplitude  K/L=kappa  UP/DOWN=attack  LEFT/RIGHT=release")
    print("          M=mode  ALT+ENTER=fullscreen  SPACE=reset  ESC=quit")
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
            viz.set_amplitude(viz.amplitude * 1.2)
        if key_ready(glfw.KEY_Z):
            viz.set_amplitude(viz.amplitude / 1.2)

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

        if key_pressed(glfw.KEY_M):
            viz.mode_type = (viz.mode_type + 1) % len(viz._mode_type_names)
            viz.print_status()

        if key_pressed(glfw.KEY_SPACE):
            viz.reset_defaults()
            cqt_proc.kappa = viz.kappa
            cqt_proc.tau_attack = viz.tau_attack
            cqt_proc.tau_release = viz.tau_release

        cqt_spectrum = audio.get_cqt()
        cqt_amps = cqt_proc.process(cqt_spectrum)
        viz.render(cqt_amps, dt)
        glfw.swap_buffers(window)

    print()
    audio.stop()
    glfw.terminate()


if __name__ == "__main__":
    main()

