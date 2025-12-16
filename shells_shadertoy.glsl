// Spectral Shell Visualizer - Constant-Q Standing Wave Synthesis
// C and Python versions: https://github.com/duffey/mvis
// IMAGE TAB - Main visualization shader
//
// Setup:
//   iChannel0 = SoundCloud/Microphone (audio input)
//   iChannel1 = Buffer A (for UI state from shells_shadertoy_buffer_a.glsl)
//
// Controls (keyboard - handled by Buffer A):
//   W/S: Scale (zoom) +/-
//   Z/X: Contrast +/-
//   K/L: Kappa (log compression) -/+
//   M: Cycle mode type (All, m!=n, Diagonal)
//   V: Cycle color mode (Plasma, Magma, Turbo, Viridis, Grayscale)
//   SPACE: Reset to defaults

#define PI 3.14159265

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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;

    // Read state from Buffer A
    vec4 state0 = texelFetch(iChannel1, ivec2(0, 0), 0);
    vec4 state1 = texelFetch(iChannel1, ivec2(1, 0), 0);

    float scale = state0.x;
    float contrast = state0.y;
    float kappa = state0.z;
    int colorMode = int(state0.w);
    int modeType = int(state1.x);

    // Fallback defaults
    if (scale < 0.01) {
        scale = 1.0;
        contrast = 1.5;
        kappa = 5.0;
        colorMode = 0;
        modeType = 0;
    }

    // Normalized coordinates
    vec2 p;
    p.x = (uv.x - 0.5) * 2.0 * aspect * scale;
    p.y = (uv.y - 0.5) * 2.0 * scale;

    // Audio parameters
    float freqStep = 11025.0 / 512.0;
    float baseFreq = 40.0;

    // Accumulate field directly from audio - no intermediate arrays
    float field = 0.0;
    float totalEnergy = 0.0;

    // For each mode (m,n), read audio at corresponding frequency and add contribution
    // Use more modes for finer detail (shells.py uses up to 32)
    for (int m = 0; m <= 20; m++) {
        for (int n = 0; n <= 20; n++) {
            if (m == 0 && n == 0) continue;

            // Mode radius determines its "frequency" in constant-Q space
            float r = sqrt(float(m*m + n*n));

            // Map mode to frequency: freq = baseFreq * 2^(s/4) where s = 4*log2(r)
            // So freq = baseFreq * r
            float freq = baseFreq * r;
            if (freq > 8000.0) continue;

            // Read FFT at this frequency
            // Shadertoy audio: 512 bins covering 0-11025 Hz
            // Values are dB scaled: 0 = -100dB, 1 = -30dB
            float u = freq / 11025.0;
            float v = texture(iChannel0, vec2(u, 0.25)).x;

            // Convert from dB back to linear amplitude
            // dB = -100 + v * 70, then amp = 10^(dB/20)
            float dB = -100.0 + v * 70.0;
            float amp = pow(10.0, dB / 20.0);

            // Apply kappa as gain (like log compression factor)
            amp *= kappa * 2.0;

            if (amp < 0.001) continue;

            float kx = float(m) * PI;
            float ky = float(n) * PI;

            float mode_val;
            if (modeType == 0) {
                mode_val = cos(kx * p.x) * cos(ky * p.y);
            } else if (modeType == 1) {
                if (m == n) continue;
                mode_val = cos(kx * p.x) * cos(ky * p.y);
            } else {
                mode_val = cos(kx * p.x) * cos(ky * p.y) + cos(ky * p.x) * cos(kx * p.y);
                if (m == n) mode_val *= 0.5;
            }

            field += amp * mode_val;
            totalEnergy += amp;
        }


    }

    // Normalize
    if (totalEnergy > 0.1) {
        field /= sqrt(totalEnergy);
    }

    // Energy rendering
    float I = field * field;
    I = pow(I, 1.0 / contrast);
    I = tanh(I * 2.0);

    // Color mapping
    vec3 color;
    if (colorMode == 0) {
        color = plasma(I);
    } else if (colorMode == 1) {
        color = magma(I);
    } else if (colorMode == 2) {
        color = turbo(I);
    } else if (colorMode == 3) {
        color = viridis(I);
    } else {
        color = vec3(I);
    }

    fragColor = vec4(color, 1.0);
}


