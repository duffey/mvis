# Audio Visualizers

Real-time audio visualizations using standing wave patterns. Captures system audio and displays animated patterns.

https://github.com/user-attachments/assets/fb3283ac-f653-423b-ba8d-fff5429fce13

## Shells

Constant-Q spectral shell visualizer using standing wave synthesis with mirror symmetry.

### How it works

1. **FFT → Shells**: Audio spectrum is grouped into logarithmically-spaced "shells" using constant-Q mapping: `s = floor(4 * log2(freq / baseFreq))`. Each shell spans a quarter-octave.

2. **Log Compression**: Shell energies are compressed: `E_s = log(1 + κ * energy)` where κ (kappa) controls sensitivity.

3. **Attack/Release Smoothing**: Shell amplitudes follow audio dynamics with configurable attack and release times.

4. **Mode Synthesis**: The field is a sum of cosine modes with even-even symmetry:
   ```
   u(x,y) = Σ A_s / N_s * cos(kx*x) * cos(ky*y)
   ```
   Each mode (m,n) belongs to shell `s = floor(4 * log2(sqrt(m² + n²)))`.

5. **Energy Rendering**: Display intensity is `I = tanh(pow(u², 1/contrast) * 2)`.

### Mode Types

- **All**: All cosine modes `cos(kx*x) * cos(ky*y)`
- **m≠n**: Only off-diagonal modes (excludes m=n)
- **Diagonal**: Adds 4-fold symmetry: `cos(kx*x)*cos(ky*y) + cos(ky*x)*cos(kx*y)`

### Files

- `shells.py` - Python implementation (requires numpy, glfw, PyOpenGL, soundcard)
- `shells.c` - C implementation for Windows (WASAPI audio, Win32 OpenGL)
- `shells_shadertoy.glsl` - Shadertoy Image shader
- `shells_shadertoy_buffer_a.glsl` - Shadertoy Buffer A (UI state)
