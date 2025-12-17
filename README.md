# Audio Visualizers

Real-time audio visualizations using standing wave patterns. Captures system audio and displays animated patterns.

## Shells

https://github.com/user-attachments/assets/291c55bc-6e10-4f02-9861-bfd1dcbd1991

CQT (Constant-Q Transform) visualizer using standing wave synthesis with mirror symmetry and water surface rendering.

### How it works

1. **CQT Analysis**: Audio is analyzed using Constant-Q Transform with 96 bins (8 octaves × 12 semitones, C1 to B8). Each bin uses a variable window length—longer for bass (better frequency resolution), shorter for treble (better time resolution).

2. **Log Compression**: CQT amplitudes are compressed: `E_k = log(1 + κ * amplitude²)` where κ (kappa) controls sensitivity.

3. **Attack/Release Smoothing**: Amplitudes follow audio dynamics with configurable attack and release times.

4. **Mode Synthesis**: Each CQT bin (musical note) drives one spatial mode with even-even symmetry:
   ```
   u(x,y) = Σ A_k * cos(kx*x) * cos(ky*y)
   ```
   Spatial frequency scales with octave; semitone patterns provide visual variety within each octave.

5. **Water Surface Rendering**: The wave field is treated as a height map. Surface normals are computed from gradients, then rendered with:
   - Fresnel reflections (edges reflect more)
   - Specular highlights
   - Height-based coloring (wave peaks glow brighter)
   - Dithering to reduce banding

### Mode Types

- **All**: All cosine modes `cos(kx*x) * cos(ky*y)`
- **m≠n**: Only off-diagonal modes (excludes m=n)
- **Diagonal**: Adds 4-fold symmetry: `cos(kx*x)*cos(ky*y) + cos(ky*x)*cos(kx*y)`

### Controls

| Key | Action |
|-----|--------|
| W/S | Scale (zoom in/out) |
| Z/X | Amplitude |
| K/L | Kappa (log compression) |
| ↑/↓ | Attack time |
| ←/→ | Release time |
| M | Cycle mode type |
| Alt+Enter | Toggle fullscreen |
| Space | Reset to defaults |
| Esc | Quit |

### Files

- `shells.py` - Python implementation (requires numpy, glfw, PyOpenGL, soundcard)
- `shells.c` - C implementation for Windows (WASAPI audio, Win32 OpenGL)
- `shells_shadertoy.glsl` - Shadertoy Image shader
- `shells_shadertoy_buffer_a.glsl` - Shadertoy Buffer A (UI state)
