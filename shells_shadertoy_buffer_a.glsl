// Spectral Shell Visualizer - Constant-Q Standing Wave Synthesis
// BUFFER A: State storage and shell amplitude processing
//
// Setup:
//   iChannel0 = Buffer A (itself, for persistence)
//   iChannel1 = Keyboard
//   iChannel2 = SoundCloud/Microphone (audio input)
//
// Pixel (0,0) stores: scale, contrast, kappa, colorMode
// Pixel (1,0) stores: modeType, tauAttack, tauRelease, unused
// Pixels (2,0) to (2+MAX_SHELLS,0): shell amplitudes A_s(t)
//
// Controls:
//   W/S: Scale (zoom) +/-
//   Z/X: Contrast +/-
//   K/L: Kappa (log compression) -/+
//   UP/DOWN: Attack time +/-
//   LEFT/RIGHT: Release time +/-
//   M: Cycle mode type (All, m!=n, Diagonal)
//   V: Cycle color mode (Plasma, Magma, Turbo, Viridis, Grayscale)
//   SPACE: Reset to defaults

#define keyClick(ascii) (texelFetch(iChannel1, ivec2(ascii, 1), 0).x > 0.)
#define keyDown(ascii)  (texelFetch(iChannel1, ivec2(ascii, 0), 0).x > 0.)

// Key codes
#define KEY_V     86
#define KEY_M     77
#define KEY_W     87
#define KEY_S     83
#define KEY_K     75
#define KEY_L     76
#define KEY_Z     90
#define KEY_X     88
#define KEY_UP    38
#define KEY_DOWN  40
#define KEY_LEFT  37
#define KEY_RIGHT 39
#define KEY_SPACE 32

// Defaults
#define DEFAULT_SCALE 1.0
#define DEFAULT_CONTRAST 1.5
#define DEFAULT_KAPPA 5.0
#define DEFAULT_COLOR_MODE 0.0   // 0=Plasma
#define DEFAULT_MODE_TYPE 0.0    // 0=All
#define DEFAULT_TAU_ATTACK 0.040
#define DEFAULT_TAU_RELEASE 0.300

#define MAX_SHELLS 32
#define BASE_FREQ 40.0

// Map frequency to shell index using constant-Q
int freqToShell(float freq) {
    if (freq < BASE_FREQ) return -1;
    float r = freq / BASE_FREQ;
    int s = int(floor(4.0 * log2(r)));
    return min(s, MAX_SHELLS - 1);
}

void mainImage(out vec4 O, in vec2 F) {
    ivec2 p = ivec2(F);

    // Only process state pixels (row 0)
    if (p.y != 0 || p.x >= 2 + MAX_SHELLS) {
        O = vec4(0);
        return;
    }

    // Load previous state
    vec4 state0 = texelFetch(iChannel0, ivec2(0, 0), 0);
    vec4 state1 = texelFetch(iChannel0, ivec2(1, 0), 0);

    // Initialize on first frame or SPACE reset
    if (iFrame == 0 || keyClick(KEY_SPACE)) {
        state0 = vec4(DEFAULT_SCALE, DEFAULT_CONTRAST, DEFAULT_KAPPA, DEFAULT_COLOR_MODE);
        state1 = vec4(DEFAULT_MODE_TYPE, DEFAULT_TAU_ATTACK, DEFAULT_TAU_RELEASE, 0.0);

        // Reset shell amplitudes
        if (p.x >= 2) {
            O = vec4(0.0);
            return;
        }
    }

    float scale = state0.x;
    float contrast = state0.y;
    float kappa = state0.z;
    float colorMode = state0.w;
    float modeType = state1.x;
    float tauAttack = state1.y;
    float tauRelease = state1.z;

    // Handle key inputs for state pixels 0 and 1
    if (p.x <= 1) {
        // V cycles color mode
        if (keyClick(KEY_V)) {
            colorMode = mod(colorMode + 1.0, 5.0);
        }
        // M cycles mode type
        if (keyClick(KEY_M)) {
            modeType = mod(modeType + 1.0, 3.0);
        }

        // Continuous adjustments
        if (keyDown(KEY_W)) scale *= 1.02;
        if (keyDown(KEY_S)) scale /= 1.02;
        if (keyDown(KEY_X)) contrast += 0.02;
        if (keyDown(KEY_Z)) contrast -= 0.02;
        if (keyDown(KEY_L)) kappa *= 1.02;
        if (keyDown(KEY_K)) kappa /= 1.02;
        if (keyDown(KEY_UP)) tauAttack *= 1.02;
        if (keyDown(KEY_DOWN)) tauAttack /= 1.02;
        if (keyDown(KEY_RIGHT)) tauRelease *= 1.02;
        if (keyDown(KEY_LEFT)) tauRelease /= 1.02;

        // Clamp parameters
        scale = clamp(scale, 0.1, 10.0);
        contrast = clamp(contrast, 0.1, 5.0);
        kappa = clamp(kappa, 0.1, 50.0);
        tauAttack = clamp(tauAttack, 0.001, 0.5);
        tauRelease = clamp(tauRelease, 0.01, 2.0);

        // Output based on pixel
        if (p.x == 0) {
            O = vec4(scale, contrast, kappa, colorMode);
        } else {
            O = vec4(modeType, tauAttack, tauRelease, 0.0);
        }
        return;
    }

    // Shell amplitude processing (pixels 2 to 2+MAX_SHELLS)
    int shellIdx = p.x - 2;
    if (shellIdx >= MAX_SHELLS) {
        O = vec4(0);
        return;
    }

    // Get previous shell amplitude
    float prevAmp = texelFetch(iChannel0, ivec2(p.x, 0), 0).x;

    // Compute raw shell energy from FFT
    float shellEnergyRaw = 0.0;
    float freqStep = 11025.0 / 512.0;  // ~21.5 Hz per bin

    for (int i = 1; i < 512; i++) {
        float freq = float(i) * freqStep;
        int s = freqToShell(freq);

        if (s == shellIdx) {
            float u = freq / 11025.0;
            float v = texture(iChannel2, vec2(u, 0.25)).x;
            // Convert from dB to linear
            float amp = pow(10.0, (-100.0 + v * 70.0) / 20.0);
            shellEnergyRaw += amp * amp;
        }
    }

    // Log compression: E_s = log(1 + κ * E_raw)
    float E_s = log(1.0 + kappa * shellEnergyRaw);

    // Attack/release smoothing
    // α = 1 - exp(-dt/τ), assuming ~60fps -> dt ≈ 0.0167
    float dt = 1.0 / 60.0;
    float alphaAttack = 1.0 - exp(-dt / tauAttack);
    float alphaRelease = 1.0 - exp(-dt / tauRelease);

    float newAmp;
    if (E_s > prevAmp) {
        newAmp = prevAmp + alphaAttack * (E_s - prevAmp);
    } else {
        newAmp = prevAmp + alphaRelease * (E_s - prevAmp);
    }

    O = vec4(newAmp, 0.0, 0.0, 1.0);
}

