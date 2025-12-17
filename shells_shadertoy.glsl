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
//   Z/X: Amplitude +/-
//   K/L: Kappa (log compression) -/+
//   UP/DOWN: Attack time +/-
//   LEFT/RIGHT: Release time +/-
//   M: Cycle mode type (All, m!=n, Diagonal)
//   SPACE: Reset to defaults
//
// Water surface lighting adapted from "Seascape" by Alexander Alekseev (TDM), 2014
// https://www.shadertoy.com/view/Ms2SD1 - CC BY-NC-SA 3.0

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
const float gradientScale = 2.0;  // Scale for gradient calculation

// Semitone patterns for mode mapping (m, n offsets for each of 12 semitones)
// These create different shapes for each note in an octave
const vec2 semitonePatterns[12] = vec2[12](
    vec2(1, 0),   // C  - horizontal
    vec2(1, 1),   // C# - diagonal
    vec2(0, 1),   // D  - vertical
    vec2(2, 1),   // D# - angled
    vec2(1, 2),   // E  - angled other way
    vec2(2, 0),   // F  - horizontal 2nd harmonic
    vec2(2, 2),   // F# - diagonal 2nd
    vec2(0, 2),   // G  - vertical 2nd
    vec2(3, 1),   // G# - complex
    vec2(1, 3),   // A  - complex other
    vec2(3, 2),   // A# - complex
    vec2(2, 3)    // B  - complex other
);

// CQT parameters
#define BINS_PER_OCTAVE 12
#define NUM_OCTAVES 8
#define NUM_CQT_BINS 96  // 8 octaves * 12 semitones
#define F_MIN 32.70      // C1 in Hz

// Compute wave field value at position p
float computeWaveField(vec2 p, int modeType, float kappa, out float totalEnergy) {
    float u = 0.0;
    totalEnergy = 0.0;

    // Process each CQT bin (musical note)
    for (int k = 0; k < NUM_CQT_BINS; k++) {
        int octave = k / BINS_PER_OCTAVE;  // 0-7
        int semitone = k - octave * BINS_PER_OCTAVE;  // 0-11

        // Base spatial frequency scales with octave
        float baseFreq = pow(2.0, float(octave) / 2.0);

        // Get pattern for this semitone
        vec2 pattern = semitonePatterns[semitone];

        // Scale by octave
        float m = max(1.0, floor(pattern.x * baseFreq));
        float n = max(0.0, floor(pattern.y * baseFreq));

        // Ensure we don't have (0,0)
        if (m < 0.5 && n < 0.5) m = 1.0;

        float kx = m * PI;
        float ky = n * PI;

        // Get amplitude for this CQT bin from audio
        // Map CQT bin to frequency: f_k = F_MIN * 2^(k/BINS_PER_OCTAVE)
        float freq = F_MIN * pow(2.0, float(k) / float(BINS_PER_OCTAVE));

        // Shadertoy audio: 512 bins covering 0-11025 Hz (at 44.1kHz sample rate)
        // Spectrum is in first row (y=0), wave in second row (y=1)
        float audioU = freq / 11025.0;
        if (audioU > 1.0) continue;

        float v = texture(iChannel0, vec2(audioU, 0.0)).x;

        // Convert from dB to linear amplitude
        float dB = -100.0 + v * 70.0;
        float amp = pow(10.0, dB / 20.0);

        // Log compression: E_k = log(1 + κ * amp²)
        float E_k = log(1.0 + kappa * amp * amp);

        if (E_k < 0.001) continue;

        float mode_val;
        if (modeType == 0) {
            // All modes
            mode_val = cos(kx * p.x) * cos(ky * p.y);
        } else if (modeType == 1) {
            // m != n only
            if (abs(m - n) < 0.5) continue;
            mode_val = cos(kx * p.x) * cos(ky * p.y);
        } else {
            // Diagonal pairs
            mode_val = cos(kx * p.x) * cos(ky * p.y) + cos(ky * p.x) * cos(kx * p.y);
            if (abs(m - n) < 0.5) mode_val *= 0.5;
        }

        u += E_k * mode_val;
        totalEnergy += E_k * E_k;
    }

    return u;
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

    // Specular with distance-dependent power
    float specPower = 600.0 * inversesqrt(max(dist * dist, 0.01));
    color += vec3(0.8, 0.9, 1.0) * specularHighlight(n, light, eye, specPower) * 1.5;

    return color;
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
    float amplitude = state0.y;
    float kappa = state0.z;
    int modeType = int(state1.x);

    // Fallback defaults
    if (scale < 0.01) {
        scale = 0.5;
        amplitude = 0.1;
        kappa = 5.0;
        modeType = 0;
    }

    // Normalized coordinates centered at origin, scaled
    vec2 p;
    p.x = (uv.x - 0.5) * 2.0 * aspect * scale;
    p.y = (uv.y - 0.5) * 2.0 * scale;

    // Small offset for gradient calculation
    float eps = 0.01 * scale;

    // Compute wave field at current position
    float totalEnergy;
    float u = computeWaveField(p, modeType, kappa, totalEnergy);

    // Compute gradient using central differences
    float dummy;
    float u_px = computeWaveField(p + vec2(eps, 0.0), modeType, kappa, dummy);
    float u_mx = computeWaveField(p - vec2(eps, 0.0), modeType, kappa, dummy);
    float u_py = computeWaveField(p + vec2(0.0, eps), modeType, kappa, dummy);
    float u_my = computeWaveField(p - vec2(0.0, eps), modeType, kappa, dummy);

    vec2 gradient = vec2(u_px - u_mx, u_py - u_my) / (2.0 * eps);
    gradient *= gradientScale;

    // Normalize by total energy
    float normFactor = amplitude / max(sqrt(totalEnergy), 0.001);
    u *= normFactor;
    gradient *= normFactor;

    // Surface normal from gradient (wave acts as height field)
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

    // Apply dithering to reduce banding
    float dither = triangularNoise(fragCoord) / 255.0;
    color += dither;

    fragColor = vec4(color, 1.0);
}
