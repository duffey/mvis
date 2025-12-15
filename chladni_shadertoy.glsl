// Chladni Plate Audio Visualizer
// Visualizes audio as vibrating plate nodal patterns
// C and Python versions available at: https://github.com/duffey/chladni

#define PI 3.14159265

// Parameters - tweak these!
#define FUNDAMENTAL 100.0
#define COMPLEXITY 0.4
#define MAX_FREQ 1100.0
#define THRESHOLD 0.1
#define COLOR_MODE 1  // 1 = color, 0 = B&W

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1., 2./3., 1./3., 3.);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6. - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0., 1.), c.y);
}

float getFFT(float freq) {
    float u = freq / 11025.;
    if (u < 0. || u > 1.) return 0.;
    float v = texture(iChannel0, vec2(u, .25)).x;
    return pow(10., (-100. + v * 70.) / 20.) * 30.;
}

float chladni(vec2 p, float n, float m, float a) {
    return cos(n*PI*p.x/a) * cos(m*PI*p.y) - cos(m*PI*p.x/a) * cos(n*PI*p.y);
}

vec2 freqToMode(float freq, float a) {
    float ratio = freq / FUNDAMENTAL;
    float l12 = 1./(a*a) + 4.;
    float target = l12 * ratio * ratio;
    if (target <= l12) return vec2(1., 2.);
    float s = sqrt(target);
    float n = max(1., floor(s * a * COMPLEXITY));
    float m = max(1., floor(sqrt(max(0., target - n*n/(a*a)))));
    if (m == n) m += 1.;
    return vec2(n, m);
}

void mainImage(out vec4 O, vec2 F) {
    float a = iResolution.x / iResolution.y;
    vec2 p = (F - .5*iResolution.xy) / (.5*iResolution.y);

    float d = 0.;
    float freqStep = 11025. / 512.;
    int maxBin = int(MAX_FREQ / freqStep);

    for (int i = 1; i < 512; i++) {
        if (i >= maxBin) break;
        float freq = float(i) * freqStep;
        float amp = getFFT(freq);
        if (amp < .001) continue;
        vec2 nm = freqToMode(freq, a);
        d += amp * chladni(p, nm.x, nm.y, a);
    }

    float line = 1. - smoothstep(THRESHOLD - fwidth(d)*1.5, THRESHOLD + fwidth(d)*1.5, abs(d));

    vec3 col = vec3(1.);
    #if COLOR_MODE
    float tot = 0., wf = 0.;
    for (int i = 1; i < 512; i++) {
        if (i >= maxBin) break;
        float freq = float(i) * freqStep;
        float amp = getFFT(freq);
        tot += amp;
        wf += freq * amp;
    }
    float avgFreq = wf / (tot + 1e-6);
    float hue = clamp(0.8 * avgFreq / MAX_FREQ, 0., 0.8);
    col = hsv2rgb(vec3(hue, 0.85, 1.));
    #endif

    O = vec4(col * line, 1.);
}
