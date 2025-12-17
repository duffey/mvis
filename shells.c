/*
 * CQT Shell Visualizer - Constant-Q Transform Standing Wave Synthesis
 * Win32 C Port
 *
 * Standing wave visualizer with mirror symmetry about x and y axes.
 * Uses even-even cosine modes: cos(kx*x) * cos(ky*y)
 *
 * Constant-Q Transform (CQT):
 *     - Frequency bins geometrically spaced (12 bins per octave = musical notes)
 *     - Variable window lengths: longer for bass, shorter for treble
 *     - Matches human perception of time-frequency tradeoff
 *     - Great for: chords, key detection, melody tracking, harmonic analysis
 *
 * Build with:
 *   cl /O2 shells.c /link opengl32.lib user32.lib gdi32.lib ole32.lib
 *
 * Or with MinGW:
 *   gcc -O2 shells.c -o shells.exe -lopengl32 -lgdi32 -lole32 -lm
 */

#define WIN32_LEAN_AND_MEAN
#define COBJMACROS
#define _USE_MATH_DEFINES

#include <windows.h>
#include <gl/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <initguid.h>
#include <mmdeviceapi.h>
#include <audioclient.h>

/* Define GUIDs for WASAPI interfaces */
DEFINE_GUID(CLSID_MMDeviceEnumerator, 0xbcde0395, 0xe52f, 0x467c,
            0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e);
DEFINE_GUID(IID_IMMDeviceEnumerator,  0xa95664d2, 0x9614, 0x4f35,
            0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6);
DEFINE_GUID(IID_IAudioClient,         0x1cb9ad4c, 0xdbfa, 0x4c32,
            0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2);
DEFINE_GUID(IID_IAudioCaptureClient,  0xc8adbd64, 0xe71e, 0x48a0,
            0xa4, 0xde, 0x18, 0x5c, 0x39, 0x5c, 0xd3, 0x17);

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "ole32.lib")

/* ============================================================================
 * OpenGL Extensions
 * ============================================================================ */

#define GL_ARRAY_BUFFER                   0x8892
#define GL_ELEMENT_ARRAY_BUFFER           0x8893
#define GL_STATIC_DRAW                    0x88E4
#define GL_DYNAMIC_DRAW                   0x88E8
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER                  0x8B31
#define GL_COMPILE_STATUS                 0x8B81
#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_TEXTURE0                       0x84C0
#define GL_TEXTURE1                       0x84C1
#define GL_TEXTURE_1D                     0x0DE0
#define GL_R32F                           0x822E
#define GL_RGBA32F                        0x8814
#define GL_CLAMP_TO_EDGE                  0x812F

typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum type);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLDELETESHADERPROC)(GLuint shader);
typedef GLint (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
typedef void (APIENTRY *PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
typedef void (APIENTRY *PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
typedef void (APIENTRY *PFNGLUNIFORM2FPROC)(GLint location, GLfloat v0, GLfloat v1);
typedef void (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
typedef void (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint array);
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY *PFNGLBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, const void *data);
typedef void (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
typedef void (APIENTRY *PFNGLACTIVETEXTUREPROC)(GLenum texture);
typedef BOOL (APIENTRY *PFNWGLSWAPINTERVALEXTPROC)(int interval);
typedef HGLRC (APIENTRY *PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC hDC, HGLRC hShareContext, const int *attribList);

#define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB      0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB  0x00000001

static PFNGLCREATESHADERPROC glCreateShader;
static PFNGLSHADERSOURCEPROC glShaderSource;
static PFNGLCOMPILESHADERPROC glCompileShader;
static PFNGLGETSHADERIVPROC glGetShaderiv;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
static PFNGLCREATEPROGRAMPROC glCreateProgram;
static PFNGLATTACHSHADERPROC glAttachShader;
static PFNGLLINKPROGRAMPROC glLinkProgram;
static PFNGLGETPROGRAMIVPROC glGetProgramiv;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
static PFNGLUSEPROGRAMPROC glUseProgram;
static PFNGLDELETESHADERPROC glDeleteShader;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
static PFNGLUNIFORM1FPROC glUniform1f;
static PFNGLUNIFORM1IPROC glUniform1i;
static PFNGLUNIFORM2FPROC glUniform2f;
static PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
static PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
static PFNGLGENBUFFERSPROC glGenBuffers;
static PFNGLBINDBUFFERPROC glBindBuffer;
static PFNGLBUFFERDATAPROC glBufferData;
static PFNGLBUFFERSUBDATAPROC glBufferSubData;
static PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
static PFNGLACTIVETEXTUREPROC glActiveTexture;
static PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
static PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;

static void LoadGLExtensions(void) {
    glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)wglGetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)wglGetProcAddress("glGetProgramInfoLog");
    glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
    glDeleteShader = (PFNGLDELETESHADERPROC)wglGetProcAddress("glDeleteShader");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
    glUniform1f = (PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f");
    glUniform1i = (PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i");
    glUniform2f = (PFNGLUNIFORM2FPROC)wglGetProcAddress("glUniform2f");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray");
    glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
    glBufferSubData = (PFNGLBUFFERSUBDATAPROC)wglGetProcAddress("glBufferSubData");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray");
    glActiveTexture = (PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture");
    wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
}

/* ============================================================================
 * Constants
 * ============================================================================ */

#define SAMPLE_RATE     44100
#define CHUNK_SIZE      64
#define MAX_FFT_SIZE    32768
#define MAX_BINS        (MAX_FFT_SIZE / 2 + 1)
#define PI              3.14159265358979323846f

/* CQT parameters */
#define BINS_PER_OCTAVE 12      /* Musical: 12 semitones per octave */
#define NUM_OCTAVES     8       /* C1 (~32 Hz) to B8 (~7902 Hz) */
#define NUM_CQT_BINS    (BINS_PER_OCTAVE * NUM_OCTAVES)  /* 96 bins total */
#define F_MIN           32.70f  /* C1 in Hz */

/* Q_FACTOR = 1 / (2^(1/BINS_PER_OCTAVE) - 1) ~= 16.82 */
#define Q_FACTOR        16.817154f

#define MAX_MODES       2048

/* ============================================================================
 * Shaders
 * ============================================================================ */

static const char *VERTEX_SHADER_SRC =
    "#version 330 core\n"
    "layout(location = 0) in vec2 position;\n"
    "out vec2 fragCoord;\n"
    "uniform vec2 iResolution;\n"
    "void main() {\n"
    "    fragCoord = (position + 1.0) * 0.5 * iResolution;\n"
    "    gl_Position = vec4(position, 0.0, 1.0);\n"
    "}\n";

static const char *FRAGMENT_SHADER_SRC =
    "#version 330 core\n"
    "in vec2 fragCoord;\n"
    "out vec4 outColor;\n"
    "\n"
    "uniform vec2 iResolution;\n"
    "uniform sampler1D iCQTAmps;         // CQT bin amplitudes (96 bins = 8 octaves x 12 semitones)\n"
    "uniform sampler1D iModeData;        // Packed mode data: (kx, ky, cqt_bin, inv_count)\n"
    "uniform int iNumModes;\n"
    "uniform int iNumCQTBins;\n"
    "uniform float iTime;\n"
    "uniform float iScale;               // Spatial frequency scale\n"
    "uniform int iModeType;              // 0=all, 1=m!=n only, 2=diagonal pairs\n"
    "uniform float iTotalEnergy;         // Total energy for normalization\n"
    "uniform float iAmplitude;           // Amplitude scaling factor\n"
    "\n"
    "#define PI 3.14159265\n"
    "\n"
    "// ============================================================================\n"
    "// DITHERING\n"
    "// ============================================================================\n"
    "\n"
    "// Hash function for pseudo-random noise\n"
    "float hash(vec2 p) {\n"
    "    vec3 p3 = fract(vec3(p.xyx) * 0.1031);\n"
    "    p3 += dot(p3, p3.yzx + 33.33);\n"
    "    return fract((p3.x + p3.y) * p3.z);\n"
    "}\n"
    "\n"
    "// Triangular dithering noise (-0.5 to 0.5, triangular distribution)\n"
    "float triangularNoise(vec2 p) {\n"
    "    float r1 = hash(p);\n"
    "    float r2 = hash(p + vec2(1.0, 0.0));\n"
    "    return (r1 + r2) * 0.5 - 0.5;\n"
    "}\n"
    "\n"
    "// ============================================================================\n"
    "// WAVE FIELD COMPUTATION\n"
    "// ============================================================================\n"
    "\n"
    "// Rendering parameters\n"
    "const float gradientScale = 2.0;        // Scale for gradient calculation\n"
    "\n"
    "// Compute wave field value at position p (normalized by total energy)\n"
    "float computeWaveField(vec2 p) {\n"
    "    float u = 0.0;\n"
    "    for (int i = 0; i < 2048; i++) {\n"
    "        if (i >= iNumModes) break;\n"
    "\n"
    "        float texCoord = (float(i) + 0.5) / float(iNumModes);\n"
    "        vec4 mode = texture(iModeData, texCoord);\n"
    "\n"
    "        float kx = mode.r;\n"
    "        float ky = mode.g;\n"
    "        int cqt_bin = int(mode.b);\n"
    "        float inv_count = mode.a;\n"
    "\n"
    "        float cqtCoord = (float(cqt_bin) + 0.5) / float(iNumCQTBins);\n"
    "        float A_s = texture(iCQTAmps, cqtCoord).r;\n"
    "\n"
    "        float mode_val;\n"
    "        if (iModeType == 0) {\n"
    "            mode_val = cos(kx * p.x) * cos(ky * p.y);\n"
    "        } else if (iModeType == 1) {\n"
    "            if (abs(kx - ky) < 0.01) continue;\n"
    "            mode_val = cos(kx * p.x) * cos(ky * p.y);\n"
    "        } else {\n"
    "            mode_val = cos(kx * p.x) * cos(ky * p.y) + cos(ky * p.x) * cos(kx * p.y);\n"
    "            if (abs(kx - ky) < 0.01) mode_val *= 0.5;\n"
    "        }\n"
    "\n"
    "        u += A_s * inv_count * mode_val;\n"
    "    }\n"
    "    // Normalize by total energy\n"
    "    return u * iAmplitude / max(sqrt(iTotalEnergy), 0.001);\n"
    "}\n"
    "\n"
    "// ============================================================================\n"
    "// LIGHTING\n"
    "// ============================================================================\n"
    "\n"
    "// Night sky background\n"
    "vec3 getSkyColor(vec3 rd) {\n"
    "    float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd) * 0.5 + 0.5;\n"
    "    sd = pow(sd, 5.0);\n"
    "    vec3 col = mix(vec3(0.05, 0.1, 0.2), vec3(0.1, 0.05, 0.2), sd);\n"
    "    return col * 0.63;\n"
    "}\n"
    "\n"
    "// Soft diffuse lighting\n"
    "float diffuse(vec3 n, vec3 l, float p) {\n"
    "    return pow(dot(n, l) * 0.4 + 0.6, p);\n"
    "}\n"
    "\n"
    "// Normalized specular (energy conserving)\n"
    "float specularHighlight(vec3 n, vec3 l, vec3 e, float s) {\n"
    "    float nrm = (s + 8.0) / (PI * 8.0);\n"
    "    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;\n"
    "}\n"
    "\n"
    "// Surface color with height-based variation\n"
    "vec3 getSurfaceColor(vec3 n, vec3 light, vec3 eye, float dist, float height) {\n"
    "    // Fresnel: edges reflect more (cubic falloff, clamped)\n"
    "    float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);\n"
    "    fresnel = min(pow(fresnel, 3.0), 0.5);\n"
    "\n"
    "    // Reflected sky\n"
    "    vec3 reflected = getSkyColor(reflect(eye, n));\n"
    "\n"
    "    // Refracted/subsurface color - darker water for specular contrast\n"
    "    vec3 baseColor = vec3(0.005, 0.01, 0.025);\n"
    "    vec3 surfaceColor = vec3(0.05, 0.1, 0.15);\n"
    "    vec3 refracted = baseColor + diffuse(n, light, 80.0) * surfaceColor * 0.05;\n"
    "\n"
    "    // Blend refracted and reflected based on fresnel\n"
    "    vec3 color = mix(refracted, reflected, fresnel);\n"
    "\n"
    "    // Distance attenuation\n"
    "    float atten = max(1.0 - dist * dist * 0.001, 0.0);\n"
    "    color += surfaceColor * atten * 0.02;\n"
    "\n"
    "    // Height-based color - wave peaks glow brighter\n"
    "    color += surfaceColor * height * 0.5 * atten;\n"
    "\n"
    "    // Specular with distance-dependent power (inversesqrt like Seascape)\n"
    "    float specPower = 600.0 * inversesqrt(max(dist * dist, 0.01));\n"
    "    color += vec3(0.8, 0.9, 1.0) * specularHighlight(n, light, eye, specPower) * 1.5;\n"
    "\n"
    "    return color;\n"
    "}\n"
    "\n"
    "// ============================================================================\n"
    "// MAIN\n"
    "// ============================================================================\n"
    "\n"
    "void main() {\n"
    "    vec2 uv = fragCoord / iResolution;\n"
    "    float aspect = iResolution.x / iResolution.y;\n"
    "\n"
    "    // Normalized coordinates centered at origin, scaled\n"
    "    vec2 p;\n"
    "    p.x = (uv.x - 0.5) * 2.0 * aspect * iScale;\n"
    "    p.y = (uv.y - 0.5) * 2.0 * iScale;\n"
    "\n"
    "    // Small offset for gradient calculation (in normalized space)\n"
    "    float eps = 0.01 * iScale;\n"
    "\n"
    "    // Compute wave field at current position\n"
    "    float u = computeWaveField(p);\n"
    "\n"
    "    // Compute gradient using central differences\n"
    "    float u_px = computeWaveField(p + vec2(eps, 0.0));\n"
    "    float u_mx = computeWaveField(p - vec2(eps, 0.0));\n"
    "    float u_py = computeWaveField(p + vec2(0.0, eps));\n"
    "    float u_my = computeWaveField(p - vec2(0.0, eps));\n"
    "\n"
    "    vec2 gradient = vec2(u_px - u_mx, u_py - u_my) / (2.0 * eps);\n"
    "    gradient *= gradientScale;\n"
    "\n"
    "    // Surface normal from gradient (wave acts as height field)\n"
    "    // Tilt slightly toward viewer for better 3D effect\n"
    "    vec3 normal = normalize(vec3(-gradient.x, -gradient.y, 1.0));\n"
    "\n"
    "    // Simulate a 3D view: eye looking down at slight angle\n"
    "    vec3 eye = normalize(vec3(uv.x - 0.5, uv.y - 0.5, -1.0));\n"
    "\n"
    "    // Light from upper-right, slightly behind viewer\n"
    "    vec3 light = normalize(vec3(0.0, 1.0, 0.8));\n"
    "\n"
    "    // Distance from center (for attenuation effects)\n"
    "    float dist = length(p);\n"
    "\n"
    "    // Get surface color with full lighting model\n"
    "    vec3 color = getSurfaceColor(normal, light, eye, dist, u);\n"
    "\n"
    "    // Apply gamma correction (linear to sRGB)\n"
    "    color = pow(color, vec3(0.65));\n"
    "\n"
    "    // Apply dithering to reduce banding (+/-0.5/255 in each channel)\n"
    "    float dither = triangularNoise(fragCoord) / 255.0;\n"
    "    color += dither;\n"
    "\n"
    "    outColor = vec4(color, 1.0);\n"
    "}\n";

/* ============================================================================
 * Audio Capture (WASAPI Loopback) with CQT
 * ============================================================================ */

typedef struct {
    float *kernelReal;      /* Real part of CQT kernel */
    float *kernelImag;      /* Imaginary part of CQT kernel */
    int windowLen;          /* Window length for this bin */
} CQTKernel;

typedef struct {
    IAudioClient *pAudioClient;
    IAudioCaptureClient *pCaptureClient;
    WAVEFORMATEX *pwfx;
    HANDLE hThread;
    volatile BOOL running;
    volatile BOOL paused;

    float ringBuffer[MAX_FFT_SIZE * 4];
    volatile LONG writePos;

    /* CQT kernels */
    CQTKernel cqtKernels[NUM_CQT_BINS];
    float cqtFreqs[NUM_CQT_BINS];
    int cqtWindowLens[NUM_CQT_BINS];

    /* Output */
    float cqtOutput[NUM_CQT_BINS];
    float runningMax;

    CRITICAL_SECTION cs;
} AudioCapture;

static AudioCapture g_audio;

static void AudioInitCQTKernels(AudioCapture *a) {
    /* Precompute CQT analysis kernels for each frequency bin.
     *
     * For each bin k:
     *     f_k = f_min * 2^(k/bins_per_octave)
     *     N_k = ceil(Q * fs / f_k)  -- Window length (longer for low freq)
     *     kernel_k = window(N_k) * exp(-2*pi*i * Q * n / N_k)
     */

    int maxWindow = 0;
    int minWindow = MAX_FFT_SIZE;

    for (int k = 0; k < NUM_CQT_BINS; k++) {
        /* Center frequency for this bin */
        float freq = F_MIN * powf(2.0f, (float)k / BINS_PER_OCTAVE);
        a->cqtFreqs[k] = freq;

        /* Window length: N_k = Q * fs / f_k */
        int N_k = (int)ceilf(Q_FACTOR * SAMPLE_RATE / freq);
        if (N_k > MAX_FFT_SIZE) N_k = MAX_FFT_SIZE;
        a->cqtWindowLens[k] = N_k;
        a->cqtKernels[k].windowLen = N_k;

        if (N_k > maxWindow) maxWindow = N_k;
        if (N_k < minWindow) minWindow = N_k;

        /* Allocate kernel */
        a->cqtKernels[k].kernelReal = (float *)malloc(N_k * sizeof(float));
        a->cqtKernels[k].kernelImag = (float *)malloc(N_k * sizeof(float));

        /* Complex exponential kernel with Hann window */
        for (int n = 0; n < N_k; n++) {
            float window = 0.5f - 0.5f * cosf(2.0f * PI * n / N_k);  /* Hann window */
            float angle = -2.0f * PI * Q_FACTOR * n / N_k;
            a->cqtKernels[k].kernelReal[n] = window * cosf(angle) / N_k;
            a->cqtKernels[k].kernelImag[n] = window * sinf(angle) / N_k;
        }
    }

    printf("CQT initialized: %d bins, %d/octave\n", NUM_CQT_BINS, BINS_PER_OCTAVE);
    printf("  Freq range: %.1f Hz - %.1f Hz\n", a->cqtFreqs[0], a->cqtFreqs[NUM_CQT_BINS - 1]);
    printf("  Window range: %d - %d samples\n", minWindow, maxWindow);
    printf("  Time resolution: %.1f - %.1f ms\n",
           (float)minWindow / SAMPLE_RATE * 1000.0f,
           (float)maxWindow / SAMPLE_RATE * 1000.0f);
}

static void AudioFreeCQTKernels(AudioCapture *a) {
    for (int k = 0; k < NUM_CQT_BINS; k++) {
        if (a->cqtKernels[k].kernelReal) free(a->cqtKernels[k].kernelReal);
        if (a->cqtKernels[k].kernelImag) free(a->cqtKernels[k].kernelImag);
    }
}

static DWORD WINAPI AudioCaptureThread(LPVOID param) {
    AudioCapture *a = (AudioCapture *)param;
    HRESULT hr;

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

    __try {
        while (a->running) {
            if (a->paused) {
                Sleep(50);
                if (a->pCaptureClient) {
                    UINT32 packetLength = 0;
                    hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);
                    while (SUCCEEDED(hr) && packetLength > 0) {
                        BYTE *pData = NULL;
                        UINT32 numFrames = 0;
                        DWORD flags = 0;
                        hr = IAudioCaptureClient_GetBuffer(a->pCaptureClient, &pData, &numFrames, &flags, NULL, NULL);
                        if (SUCCEEDED(hr)) {
                            IAudioCaptureClient_ReleaseBuffer(a->pCaptureClient, numFrames);
                        }
                        hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);
                    }
                }
                continue;
            }

            if (!a->pCaptureClient) break;

            UINT32 packetLength = 0;
            hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);

            if (SUCCEEDED(hr) && packetLength > 0) {
                while (SUCCEEDED(hr) && packetLength > 0 && a->running && !a->paused) {
                    BYTE *pData = NULL;
                    UINT32 numFrames = 0;
                    DWORD flags = 0;

                    hr = IAudioCaptureClient_GetBuffer(a->pCaptureClient, &pData, &numFrames, &flags, NULL, NULL);
                    if (FAILED(hr) || !pData) break;

                    float *samples = (float *)pData;
                    int channels = a->pwfx->nChannels;
                    int bufLen = MAX_FFT_SIZE * 4;

                    EnterCriticalSection(&a->cs);
                    for (UINT32 i = 0; i < numFrames; i++) {
                        float sample = 0.0f;
                        for (int c = 0; c < channels; c++) {
                            sample += samples[i * channels + c];
                        }
                        sample /= channels;

                        int pos = a->writePos % bufLen;
                        a->ringBuffer[pos] = sample;
                        a->writePos = (a->writePos + 1) % bufLen;
                    }
                    LeaveCriticalSection(&a->cs);

                    hr = IAudioCaptureClient_ReleaseBuffer(a->pCaptureClient, numFrames);
                    if (FAILED(hr)) break;
                    hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);
                }
            } else {
                Sleep(1);
            }
        }
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        printf("\nAudio thread crashed!\n");
        fflush(stdout);
    }

    return 0;
}

static BOOL AudioInit(AudioCapture *a) {
    HRESULT hr;
    IMMDeviceEnumerator *pEnumerator = NULL;
    IMMDevice *pDevice = NULL;

    memset(a, 0, sizeof(*a));
    a->runningMax = 0.01f;
    a->paused = FALSE;
    InitializeCriticalSection(&a->cs);

    /* Initialize CQT kernels */
    AudioInitCQTKernels(a);

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != S_FALSE && hr != RPC_E_CHANGED_MODE) {
        printf("CoInitializeEx failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &IID_IMMDeviceEnumerator, (void **)&pEnumerator);
    if (FAILED(hr)) {
        printf("CoCreateInstance failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(pEnumerator, eRender, eConsole, &pDevice);
    IMMDeviceEnumerator_Release(pEnumerator);
    if (FAILED(hr)) {
        printf("GetDefaultAudioEndpoint failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IMMDevice_Activate(pDevice, &IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&a->pAudioClient);
    IMMDevice_Release(pDevice);
    if (FAILED(hr)) {
        printf("Activate failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IAudioClient_GetMixFormat(a->pAudioClient, &a->pwfx);
    if (FAILED(hr)) {
        printf("GetMixFormat failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IAudioClient_Initialize(a->pAudioClient, AUDCLNT_SHAREMODE_SHARED,
                                  AUDCLNT_STREAMFLAGS_LOOPBACK,
                                  10000000, 0, a->pwfx, NULL);
    if (FAILED(hr)) {
        printf("Initialize failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IAudioClient_GetService(a->pAudioClient, &IID_IAudioCaptureClient, (void **)&a->pCaptureClient);
    if (FAILED(hr)) {
        printf("GetService failed: 0x%08lx\n", hr);
        return FALSE;
    }

    a->running = TRUE;
    a->hThread = CreateThread(NULL, 0, AudioCaptureThread, a, 0, NULL);

    hr = IAudioClient_Start(a->pAudioClient);
    if (FAILED(hr)) {
        printf("Start failed: 0x%08lx\n", hr);
        return FALSE;
    }

    printf("Audio initialized: %d Hz, %d channels\n", a->pwfx->nSamplesPerSec, a->pwfx->nChannels);
    return TRUE;
}

static void AudioStop(AudioCapture *a) {
    a->running = FALSE;
    if (a->hThread) {
        WaitForSingleObject(a->hThread, 1000);
        CloseHandle(a->hThread);
    }
    if (a->pAudioClient) IAudioClient_Stop(a->pAudioClient);
    if (a->pCaptureClient) IAudioCaptureClient_Release(a->pCaptureClient);
    if (a->pAudioClient) IAudioClient_Release(a->pAudioClient);
    if (a->pwfx) CoTaskMemFree(a->pwfx);
    AudioFreeCQTKernels(a);
    DeleteCriticalSection(&a->cs);
}

static float *AudioGetCQT(AudioCapture *a) {
    /* Compute Constant-Q Transform using precomputed kernels.
     *
     * Each bin uses a different window length, providing:
     * - Better frequency resolution for bass (long windows)
     * - Better time resolution for treble (short windows)
     */
    int bufLen = MAX_FFT_SIZE * 4;

    EnterCriticalSection(&a->cs);
    int currentPos = a->writePos;
    LeaveCriticalSection(&a->cs);

    for (int k = 0; k < NUM_CQT_BINS; k++) {
        int N_k = a->cqtKernels[k].windowLen;
        float *kernelReal = a->cqtKernels[k].kernelReal;
        float *kernelImag = a->cqtKernels[k].kernelImag;

        /* Extract samples for this bin's window length */
        int start = (currentPos - N_k + bufLen) % bufLen;

        /* Convolve with kernel (dot product for matched frequency) */
        float sumReal = 0.0f;
        float sumImag = 0.0f;

        EnterCriticalSection(&a->cs);
        for (int n = 0; n < N_k; n++) {
            int idx = (start + n) % bufLen;
            float sample = a->ringBuffer[idx];
            sumReal += sample * kernelReal[n];
            sumImag += sample * kernelImag[n];
        }
        LeaveCriticalSection(&a->cs);

        a->cqtOutput[k] = sqrtf(sumReal * sumReal + sumImag * sumImag);
    }

    /* Adaptive normalization with slower response and much higher floor
     * This prevents over-saturation at loud volumes */
    float currentMax = 0.0f;
    for (int k = 0; k < NUM_CQT_BINS; k++) {
        if (a->cqtOutput[k] > currentMax) currentMax = a->cqtOutput[k];
    }

    /* Slow attack (don't jump up instantly), slower release */
    if (currentMax > a->runningMax) {
        /* Very slow attack */
        a->runningMax += 0.01f * (currentMax - a->runningMax);
    } else {
        /* Very slow release */
        a->runningMax = a->runningMax * 0.999f;
        if (a->runningMax < currentMax) a->runningMax = currentMax;
        if (a->runningMax < 0.0001f) a->runningMax = 0.0001f;
    }

    /* Use much higher reference point to leave lots of headroom
     * 5x headroom means normal volume sits around 20% amplitude */
    float reference = a->runningMax * 5.0f + 0.0001f;

    for (int k = 0; k < NUM_CQT_BINS; k++) {
        float normalized = a->cqtOutput[k] / reference;
        if (normalized > 1.0f) normalized = 1.0f;
        if (normalized < 0.0f) normalized = 0.0f;
        a->cqtOutput[k] = normalized;
    }

    return a->cqtOutput;
}

/* ============================================================================
 * CQT Processor - Maps CQT bins to spatial modes
 * ============================================================================ */

typedef struct {
    int numModes;
    float kappa;
    float tauAttack;
    float tauRelease;

    float modeData[MAX_MODES * 4];  /* (kx, ky, cqt_bin, inv_count) */
    float cqtAmps[NUM_CQT_BINS];

    LARGE_INTEGER lastTime;
    LARGE_INTEGER freq;
} CQTProcessor;

static CQTProcessor g_cqt;

/* Semitone patterns for the 12 notes (gives visual variety) */
static const int semitone_patterns[12][2] = {
    {1, 0},   /* C  - horizontal */
    {1, 1},   /* C# - diagonal */
    {0, 1},   /* D  - vertical */
    {2, 1},   /* D# - angled */
    {1, 2},   /* E  - angled other way */
    {2, 0},   /* F  - horizontal 2nd harmonic */
    {2, 2},   /* F# - diagonal 2nd */
    {0, 2},   /* G  - vertical 2nd */
    {3, 1},   /* G# - complex */
    {1, 3},   /* A  - complex other */
    {3, 2},   /* A# - complex */
    {2, 3},   /* B  - complex other */
};

static void CQTProcessorInit(CQTProcessor *cp) {
    memset(cp, 0, sizeof(*cp));
    cp->kappa = 1.0f;           /* Lower = more dynamic range */
    cp->tauAttack = 0.040f;
    cp->tauRelease = 0.300f;

    QueryPerformanceFrequency(&cp->freq);
    QueryPerformanceCounter(&cp->lastTime);

    /* Generate modes - one per CQT bin
     * Much simpler mapping: each musical note gets ONE distinct spatial mode.
     * Lower notes = lower spatial frequency, higher notes = higher spatial frequency.
     */
    int modeIdx = 0;

    for (int k = 0; k < NUM_CQT_BINS && modeIdx < MAX_MODES; k++) {
        int octave = k / BINS_PER_OCTAVE;     /* 0-7 */
        int semitone = k % BINS_PER_OCTAVE;   /* 0-11 */

        /* Base spatial frequency scales with octave (doubles each octave) */
        float base_freq = powf(2.0f, octave / 2.0f);  /* Slower growth for visibility */

        /* Get pattern for this semitone */
        int m_base = semitone_patterns[semitone][0];
        int n_base = semitone_patterns[semitone][1];

        /* Scale by octave */
        int m = (int)(m_base * base_freq);
        int n = (int)(n_base * base_freq);
        if (m < 1 && m_base > 0) m = 1;
        if (n < 0) n = 0;

        /* Ensure we don't have (0,0) */
        if (m == 0 && n == 0) m = 1;

        float kx = m * PI;
        float ky = n * PI;

        cp->modeData[modeIdx * 4 + 0] = kx;
        cp->modeData[modeIdx * 4 + 1] = ky;
        cp->modeData[modeIdx * 4 + 2] = (float)k;   /* CQT bin index */
        cp->modeData[modeIdx * 4 + 3] = 1.0f;       /* Each bin has exactly one mode */
        modeIdx++;
    }
    cp->numModes = modeIdx;

    printf("Simple mode mapping: %d modes (1 per CQT bin)\n", cp->numModes);
}

static void CQTProcessorProcess(CQTProcessor *cp, float *cqtSpectrum) {
    /* Process CQT spectrum into smoothed amplitudes with attack/release dynamics */
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    float dt = (float)(now.QuadPart - cp->lastTime.QuadPart) / (float)cp->freq.QuadPart;
    cp->lastTime = now;

    /* Attack/release smoothing */
    float alphaAttack = 1.0f - expf(-dt / cp->tauAttack);
    float alphaRelease = 1.0f - expf(-dt / cp->tauRelease);

    for (int k = 0; k < NUM_CQT_BINS; k++) {
        /* Log compression: E_k = log(1 + kappa * spectrum^2) */
        float E_k = logf(1.0f + cp->kappa * cqtSpectrum[k] * cqtSpectrum[k]);

        /* A_k <- A_k + alpha * (E_k - A_k) */
        if (E_k > cp->cqtAmps[k]) {
            cp->cqtAmps[k] += alphaAttack * (E_k - cp->cqtAmps[k]);
        } else {
            cp->cqtAmps[k] += alphaRelease * (E_k - cp->cqtAmps[k]);
        }
    }
}

/* ============================================================================
 * Visualizer
 * ============================================================================ */

#define DEFAULT_SCALE        0.5f
#define DEFAULT_KAPPA        1.0f
#define DEFAULT_TAU_ATTACK   0.040f
#define DEFAULT_TAU_RELEASE  0.300f
#define DEFAULT_AMPLITUDE    0.1f
#define DEFAULT_MODE_TYPE    0

#define NUM_MODE_TYPES       3

static const char *MODE_TYPE_NAMES[] = {"All", "m!=n", "Diagonal"};

typedef struct {
    int w, h;
    float scale;
    int modeType;
    float amplitude;
    float time;

    GLuint program;
    GLuint vao;
    GLuint cqtTex;
    GLuint modeTex;

    GLint locResolution;
    GLint locCQTAmps;
    GLint locModeData;
    GLint locNumModes;
    GLint locNumCQTBins;
    GLint locTime;
    GLint locScale;
    GLint locModeType;
    GLint locTotalEnergy;
    GLint locAmplitude;
} Visualizer;

static Visualizer g_viz;

static GLuint CompileShader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, NULL, log);
        printf("Shader compile error: %s\n", log);
        return 0;
    }
    return shader;
}

static void VizResetDefaults(Visualizer *v) {
    v->scale = DEFAULT_SCALE;
    v->modeType = DEFAULT_MODE_TYPE;
    v->amplitude = DEFAULT_AMPLITUDE;
}

static BOOL VizInit(Visualizer *v, int w, int h, CQTProcessor *cp) {
    v->w = w;
    v->h = h;
    v->time = 0.0f;
    VizResetDefaults(v);

    GLuint vs = CompileShader(GL_VERTEX_SHADER, VERTEX_SHADER_SRC);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
    if (!vs || !fs) return FALSE;

    v->program = glCreateProgram();
    glAttachShader(v->program, vs);
    glAttachShader(v->program, fs);
    glLinkProgram(v->program);

    GLint status;
    glGetProgramiv(v->program, GL_LINK_STATUS, &status);
    if (!status) {
        char log[1024];
        glGetProgramInfoLog(v->program, 1024, NULL, log);
        printf("Program link error: %s\n", log);
        return FALSE;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    float verts[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    unsigned int inds[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &v->vao);
    GLuint vbo, ebo;
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(v->vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(inds), inds, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);

    /* 1D texture for CQT amplitudes (96 bins) */
    glGenTextures(1, &v->cqtTex);
    glBindTexture(GL_TEXTURE_1D, v->cqtTex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NUM_CQT_BINS, 0, GL_RED, GL_FLOAT, NULL);

    /* 1D texture for mode data (RGBA: kx, ky, cqt_bin, inv_count) */
    glGenTextures(1, &v->modeTex);
    glBindTexture(GL_TEXTURE_1D, v->modeTex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, cp->numModes, 0, GL_RGBA, GL_FLOAT, cp->modeData);

    v->locResolution = glGetUniformLocation(v->program, "iResolution");
    v->locCQTAmps = glGetUniformLocation(v->program, "iCQTAmps");
    v->locModeData = glGetUniformLocation(v->program, "iModeData");
    v->locNumModes = glGetUniformLocation(v->program, "iNumModes");
    v->locNumCQTBins = glGetUniformLocation(v->program, "iNumCQTBins");
    v->locTime = glGetUniformLocation(v->program, "iTime");
    v->locScale = glGetUniformLocation(v->program, "iScale");
    v->locModeType = glGetUniformLocation(v->program, "iModeType");
    v->locTotalEnergy = glGetUniformLocation(v->program, "iTotalEnergy");
    v->locAmplitude = glGetUniformLocation(v->program, "iAmplitude");

    return TRUE;
}

static void VizRender(Visualizer *v, CQTProcessor *cp, float dt) {
    v->time += dt;

    /* Compute total energy (sum of squared amplitudes) */
    float totalEnergy = 0.0f;
    for (int k = 0; k < NUM_CQT_BINS; k++) {
        totalEnergy += cp->cqtAmps[k] * cp->cqtAmps[k];
    }

    /* Update CQT amplitudes texture */
    glBindTexture(GL_TEXTURE_1D, v->cqtTex);
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NUM_CQT_BINS, GL_RED, GL_FLOAT, cp->cqtAmps);

    glViewport(0, 0, v->w, v->h);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(v->program);
    glUniform2f(v->locResolution, (float)v->w, (float)v->h);
    glUniform1i(v->locNumModes, cp->numModes);
    glUniform1i(v->locNumCQTBins, NUM_CQT_BINS);
    glUniform1f(v->locTime, v->time);
    glUniform1f(v->locScale, v->scale);
    glUniform1i(v->locModeType, v->modeType);
    glUniform1f(v->locTotalEnergy, totalEnergy);
    glUniform1f(v->locAmplitude, v->amplitude);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, v->cqtTex);
    glUniform1i(v->locCQTAmps, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, v->modeTex);
    glUniform1i(v->locModeData, 1);

    glBindVertexArray(v->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);
}

static void PrintStatus(Visualizer *v, CQTProcessor *cp) {
    char status[256];
    snprintf(status, sizeof(status),
             "Scale=%.2f  Amp=%.2f  Kappa=%.1f  Attack=%.0fms  Release=%.0fms  Mode=%s",
             v->scale, v->amplitude, cp->kappa,
             cp->tauAttack * 1000.0f, cp->tauRelease * 1000.0f,
             MODE_TYPE_NAMES[v->modeType]);
    printf("\r%-100s", status);
    fflush(stdout);
}

/* ============================================================================
 * Win32 Window
 * ============================================================================ */

static HWND g_hwnd = NULL;
static HDC g_hdc;
static HGLRC g_hglrc;
static BOOL g_fullscreen = FALSE;
static RECT g_windowedRect;
static DWORD g_windowedStyle;

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED && hwnd == g_hwnd && g_hglrc) {
                g_viz.w = LOWORD(lParam);
                g_viz.h = HIWORD(lParam);
            }
            break;
        case WM_DESTROY:
            if (hwnd == g_hwnd) {
                PostQuitMessage(0);
            }
            return 0;
        case WM_CLOSE:
            if (hwnd == g_hwnd) {
                PostQuitMessage(0);
            }
            return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static void ToggleFullscreen(void) {
    if (g_fullscreen) {
        SetWindowLong(g_hwnd, GWL_STYLE, g_windowedStyle);
        SetWindowPos(g_hwnd, NULL, g_windowedRect.left, g_windowedRect.top,
                     g_windowedRect.right - g_windowedRect.left,
                     g_windowedRect.bottom - g_windowedRect.top,
                     SWP_FRAMECHANGED | SWP_NOZORDER);
        g_fullscreen = FALSE;
    } else {
        g_windowedStyle = GetWindowLong(g_hwnd, GWL_STYLE);
        GetWindowRect(g_hwnd, &g_windowedRect);

        MONITORINFO mi = { sizeof(mi) };
        GetMonitorInfo(MonitorFromWindow(g_hwnd, MONITOR_DEFAULTTOPRIMARY), &mi);
        SetWindowLong(g_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(g_hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                     mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top,
                     SWP_FRAMECHANGED);
        g_fullscreen = TRUE;
    }
}

static BOOL CreateOpenGLContext(int w, int h) {
    WNDCLASS wc = {0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "ShellsClass";
    RegisterClass(&wc);

    HWND tempHwnd = CreateWindow("ShellsClass", "temp", WS_POPUP, 0, 0, 1, 1, NULL, NULL, wc.hInstance, NULL);
    HDC tempDC = GetDC(tempHwnd);

    PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR), 1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA, 32,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        24, 8, 0, PFD_MAIN_PLANE, 0, 0, 0, 0
    };

    int pf = ChoosePixelFormat(tempDC, &pfd);
    SetPixelFormat(tempDC, pf, &pfd);
    HGLRC tempRC = wglCreateContext(tempDC);
    wglMakeCurrent(tempDC, tempRC);

    LoadGLExtensions();

    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(tempRC);
    ReleaseDC(tempHwnd, tempDC);
    DestroyWindow(tempHwnd);

    RECT rect = { 0, 0, w, h };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    int winW = rect.right - rect.left;
    int winH = rect.bottom - rect.top;

    g_hwnd = CreateWindow("ShellsClass", "CQT Shell Visualizer", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                          (screenW - winW) / 2, (screenH - winH) / 2, winW, winH,
                          NULL, NULL, wc.hInstance, NULL);
    if (!g_hwnd) return FALSE;

    g_hdc = GetDC(g_hwnd);
    pf = ChoosePixelFormat(g_hdc, &pfd);
    SetPixelFormat(g_hdc, pf, &pfd);

    if (wglCreateContextAttribsARB) {
        int attribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
            WGL_CONTEXT_MINOR_VERSION_ARB, 3,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            0
        };
        g_hglrc = wglCreateContextAttribsARB(g_hdc, NULL, attribs);
    } else {
        g_hglrc = wglCreateContext(g_hdc);
    }

    if (!g_hglrc) return FALSE;
    wglMakeCurrent(g_hdc, g_hglrc);

    if (wglSwapIntervalEXT) {
        wglSwapIntervalEXT(1);
    }

    return TRUE;
}

static void DestroyOpenGLContext(void) {
    wglMakeCurrent(NULL, NULL);
    if (g_hglrc) wglDeleteContext(g_hglrc);
    if (g_hdc) ReleaseDC(g_hwnd, g_hdc);
    if (g_hwnd) DestroyWindow(g_hwnd);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char *argv[]) {
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

    int winW = 1280, winH = 720;

    if (!CreateOpenGLContext(winW, winH)) {
        printf("Failed to create OpenGL context\n");
        return 1;
    }

    CQTProcessorInit(&g_cqt);

    if (!VizInit(&g_viz, winW, winH, &g_cqt)) {
        printf("Failed to initialize visualizer\n");
        DestroyOpenGLContext();
        return 1;
    }

    if (!AudioInit(&g_audio)) {
        printf("Failed to initialize audio capture\n");
        DestroyOpenGLContext();
        return 1;
    }

    Sleep(50);

    printf("CQT Shell Visualizer - Constant-Q Transform Standing Wave Synthesis\n");
    printf("Controls: W/S=scale  Z/X=amplitude  K/L=kappa  UP/DOWN=attack  LEFT/RIGHT=release\n");
    printf("          M=mode  ALT+ENTER=fullscreen  SPACE=reset  ESC=quit\n\n");

    PrintStatus(&g_viz, &g_cqt);

    BOOL running = TRUE;
    DWORD lastKeyTime = 0;
    DWORD repeatDelay = 100;
    BOOL prevM = FALSE, prevSpace = FALSE;
    BOOL prevAltEnter = FALSE;

    LARGE_INTEGER perfFreq, lastTime, nowTime;
    QueryPerformanceFrequency(&perfFreq);
    QueryPerformanceCounter(&lastTime);

    while (running) {
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = FALSE;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!running) break;

        QueryPerformanceCounter(&nowTime);
        float dt = (float)(nowTime.QuadPart - lastTime.QuadPart) / (float)perfFreq.QuadPart;
        lastTime = nowTime;

        DWORD now = GetTickCount();
        BOOL needUpdate = FALSE;

        BOOL hasFocus = (GetForegroundWindow() == g_hwnd);

        if (hasFocus) {
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
                printf("\nESC pressed - exiting\n");
                fflush(stdout);
                running = FALSE;
                continue;
            }

            BOOL altHeld = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
            BOOL enterPressed = (GetAsyncKeyState(VK_RETURN) & 0x8000) != 0;
            BOOL currAltEnter = altHeld && enterPressed;
            if (currAltEnter && !prevAltEnter) {
                ToggleFullscreen();
            }
            prevAltEnter = currAltEnter;

            /* Scale W/S */
            if ((GetAsyncKeyState('W') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.scale *= 1.1f;
                if (g_viz.scale > 10.0f) g_viz.scale = 10.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('S') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.scale /= 1.1f;
                if (g_viz.scale < 0.1f) g_viz.scale = 0.1f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Amplitude Z/X */
            if ((GetAsyncKeyState('X') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.amplitude *= 1.2f;
                if (g_viz.amplitude > 5.0f) g_viz.amplitude = 5.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('Z') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.amplitude /= 1.2f;
                if (g_viz.amplitude < 0.01f) g_viz.amplitude = 0.01f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Kappa K/L */
            if ((GetAsyncKeyState('L') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_cqt.kappa *= 1.2f;
                if (g_cqt.kappa > 50.0f) g_cqt.kappa = 50.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('K') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_cqt.kappa /= 1.2f;
                if (g_cqt.kappa < 0.1f) g_cqt.kappa = 0.1f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Attack UP/DOWN */
            if ((GetAsyncKeyState(VK_UP) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_cqt.tauAttack *= 1.2f;
                if (g_cqt.tauAttack > 0.5f) g_cqt.tauAttack = 0.5f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_DOWN) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_cqt.tauAttack /= 1.2f;
                if (g_cqt.tauAttack < 0.001f) g_cqt.tauAttack = 0.001f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Release LEFT/RIGHT */
            if ((GetAsyncKeyState(VK_RIGHT) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_cqt.tauRelease *= 1.2f;
                if (g_cqt.tauRelease > 2.0f) g_cqt.tauRelease = 2.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_LEFT) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_cqt.tauRelease /= 1.2f;
                if (g_cqt.tauRelease < 0.01f) g_cqt.tauRelease = 0.01f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Mode type M (edge-triggered) */
            BOOL currM = (GetAsyncKeyState('M') & 0x8000) != 0;
            if (currM && !prevM) {
                g_viz.modeType = (g_viz.modeType + 1) % NUM_MODE_TYPES;
                needUpdate = TRUE;
            }
            prevM = currM;

            /* Reset SPACE (edge-triggered) */
            BOOL currSpace = (GetAsyncKeyState(VK_SPACE) & 0x8000) != 0;
            if (currSpace && !prevSpace) {
                VizResetDefaults(&g_viz);
                g_cqt.kappa = DEFAULT_KAPPA;
                g_cqt.tauAttack = DEFAULT_TAU_ATTACK;
                g_cqt.tauRelease = DEFAULT_TAU_RELEASE;
                needUpdate = TRUE;
            }
            prevSpace = currSpace;
        } else {
            prevM = FALSE;
            prevSpace = FALSE;
            prevAltEnter = FALSE;
        }

        /* Update viewport if window was resized */
        static int lastW = 0, lastH = 0;
        if (g_viz.w != lastW || g_viz.h != lastH) {
            if (g_viz.w > 0 && g_viz.h > 0) {
                glViewport(0, 0, g_viz.w, g_viz.h);
                lastW = g_viz.w;
                lastH = g_viz.h;
            }
        }

        if (g_viz.w <= 0 || g_viz.h <= 0) {
            Sleep(16);
            continue;
        }

        /* Render */
        if (g_hdc && wglGetCurrentContext()) {
            float *cqtSpectrum = AudioGetCQT(&g_audio);
            if (cqtSpectrum) {
                CQTProcessorProcess(&g_cqt, cqtSpectrum);
                VizRender(&g_viz, &g_cqt, dt);
                SwapBuffers(g_hdc);

                if (needUpdate) {
                    PrintStatus(&g_viz, &g_cqt);
                }
            }
        } else {
            Sleep(16);
        }
    }

    printf("\n");
    AudioStop(&g_audio);
    DestroyOpenGLContext();
    CoUninitialize();

    return 0;
}
