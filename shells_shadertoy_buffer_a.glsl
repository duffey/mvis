// Spectral Shell Visualizer - Constant-Q Standing Wave Synthesis
// BUFFER A: State storage for UI parameters
//
// Setup:
//   iChannel0 = Buffer A (itself, for persistence)
//   iChannel1 = Keyboard
//
// Pixel (0,0) stores: scale, amplitude, kappa, unused
// Pixel (1,0) stores: modeType, tauAttack, tauRelease, unused
//
// Controls:
//   W/S: Scale (zoom) +/-
//   Z/X: Amplitude +/-
//   K/L: Kappa (log compression) -/+
//   UP/DOWN: Attack time +/-
//   LEFT/RIGHT: Release time +/-
//   M: Cycle mode type (All, m!=n, Diagonal)
//   SPACE: Reset to defaults

#define keyClick(ascii) (texelFetch(iChannel1, ivec2(ascii, 1), 0).x > 0.)
#define keyDown(ascii)  (texelFetch(iChannel1, ivec2(ascii, 0), 0).x > 0.)

// Key codes
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
#define DEFAULT_SCALE 0.5
#define DEFAULT_AMPLITUDE 0.1
#define DEFAULT_KAPPA 5.0
#define DEFAULT_MODE_TYPE 0.0    // 0=All
#define DEFAULT_TAU_ATTACK 0.040
#define DEFAULT_TAU_RELEASE 0.300

void mainImage(out vec4 O, in vec2 F) {
    ivec2 p = ivec2(F);

    // Only process state pixels (row 0, columns 0-1)
    if (p.y != 0 || p.x >= 2) {
        O = vec4(0);
        return;
    }

    // Load previous state
    vec4 state0 = texelFetch(iChannel0, ivec2(0, 0), 0);
    vec4 state1 = texelFetch(iChannel0, ivec2(1, 0), 0);

    // Initialize on first frame or SPACE reset
    if (iFrame == 0 || keyClick(KEY_SPACE)) {
        state0 = vec4(DEFAULT_SCALE, DEFAULT_AMPLITUDE, DEFAULT_KAPPA, 0.0);
        state1 = vec4(DEFAULT_MODE_TYPE, DEFAULT_TAU_ATTACK, DEFAULT_TAU_RELEASE, 0.0);
    }

    float scale = state0.x;
    float amplitude = state0.y;
    float kappa = state0.z;
    float modeType = state1.x;
    float tauAttack = state1.y;
    float tauRelease = state1.z;

    // M cycles mode type
    if (keyClick(KEY_M)) {
        modeType = mod(modeType + 1.0, 3.0);
    }

    // Continuous adjustments
    if (keyDown(KEY_W)) scale *= 1.02;
    if (keyDown(KEY_S)) scale /= 1.02;
    if (keyDown(KEY_X)) amplitude *= 1.02;
    if (keyDown(KEY_Z)) amplitude /= 1.02;
    if (keyDown(KEY_L)) kappa *= 1.02;
    if (keyDown(KEY_K)) kappa /= 1.02;
    if (keyDown(KEY_UP)) tauAttack *= 1.02;
    if (keyDown(KEY_DOWN)) tauAttack /= 1.02;
    if (keyDown(KEY_RIGHT)) tauRelease *= 1.02;
    if (keyDown(KEY_LEFT)) tauRelease /= 1.02;

    // Clamp parameters (matching Python ranges)
    scale = clamp(scale, 0.1, 10.0);
    amplitude = clamp(amplitude, 0.01, 5.0);
    kappa = clamp(kappa, 0.1, 50.0);
    tauAttack = clamp(tauAttack, 0.001, 0.5);
    tauRelease = clamp(tauRelease, 0.01, 2.0);

    // Output based on pixel
    if (p.x == 0) {
        O = vec4(scale, amplitude, kappa, 0.0);
    } else {
        O = vec4(modeType, tauAttack, tauRelease, 0.0);
    }
}
