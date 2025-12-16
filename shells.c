/*
 * Spectral Shell Visualizer - Constant-Q Standing Wave Synthesis
 * Win32 C Port
 *
 * Standing wave visualizer with mirror symmetry about x and y axes.
 * Uses even-even cosine modes: cos(kx*x) * cos(ky*y)
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

#define MAX_MODES       2048
#define MAX_SHELLS      64

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
    "uniform sampler1D iShellAmps;\n"
    "uniform sampler1D iModeData;\n"
    "uniform int iNumModes;\n"
    "uniform int iNumShells;\n"
    "uniform float iTime;\n"
    "uniform int iColorMode;\n"
    "uniform float iContrast;\n"
    "uniform float iScale;\n"
    "uniform int iModeType;\n"
    "\n"
    "#define PI 3.14159265\n"
    "#define TAU 6.28318530\n"
    "\n"
    "// ============================================================================\n"
    "// COLORMAPS\n"
    "// ============================================================================\n"
    "\n"
    "vec3 plasma(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.050383, 0.029803, 0.527975);\n"
    "    vec3 c1 = vec3(0.417642, 0.000564, 0.658390);\n"
    "    vec3 c2 = vec3(0.692840, 0.165141, 0.564522);\n"
    "    vec3 c3 = vec3(0.881443, 0.392529, 0.383229);\n"
    "    vec3 c4 = vec3(0.987622, 0.645320, 0.039886);\n"
    "    vec3 c5 = vec3(0.940015, 0.975158, 0.131326);\n"
    "    if (t >= 1.0) return c5;\n"
    "    float s = t * 5.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    return mix(c4, c5, f);\n"
    "}\n"
    "\n"
    "vec3 magma(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.001462, 0.000466, 0.013866);\n"
    "    vec3 c1 = vec3(0.316654, 0.071862, 0.485380);\n"
    "    vec3 c2 = vec3(0.716387, 0.214982, 0.474720);\n"
    "    vec3 c3 = vec3(0.974417, 0.462840, 0.359756);\n"
    "    vec3 c4 = vec3(0.995131, 0.766837, 0.534094);\n"
    "    vec3 c5 = vec3(0.987053, 0.991438, 0.749504);\n"
    "    if (t >= 1.0) return c5;\n"
    "    float s = t * 5.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    return mix(c4, c5, f);\n"
    "}\n"
    "\n"
    "vec3 turbo(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.18995, 0.07176, 0.23217);\n"
    "    vec3 c1 = vec3(0.25107, 0.25237, 0.63374);\n"
    "    vec3 c2 = vec3(0.15992, 0.53830, 0.72889);\n"
    "    vec3 c3 = vec3(0.09140, 0.74430, 0.54318);\n"
    "    vec3 c4 = vec3(0.52876, 0.85393, 0.21546);\n"
    "    vec3 c5 = vec3(0.88092, 0.73551, 0.07741);\n"
    "    vec3 c6 = vec3(0.97131, 0.45935, 0.05765);\n"
    "    vec3 c7 = vec3(0.84299, 0.15070, 0.15090);\n"
    "    if (t >= 1.0) return c7;\n"
    "    float s = t * 7.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    if (idx == 4) return mix(c4, c5, f);\n"
    "    if (idx == 5) return mix(c5, c6, f);\n"
    "    return mix(c6, c7, f);\n"
    "}\n"
    "\n"
    "vec3 viridis(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.267004, 0.004874, 0.329415);\n"
    "    vec3 c1 = vec3(0.282327, 0.140926, 0.457517);\n"
    "    vec3 c2 = vec3(0.253935, 0.265254, 0.529983);\n"
    "    vec3 c3 = vec3(0.206756, 0.371758, 0.553117);\n"
    "    vec3 c4 = vec3(0.143936, 0.522773, 0.556295);\n"
    "    vec3 c5 = vec3(0.119512, 0.607464, 0.540218);\n"
    "    vec3 c6 = vec3(0.166383, 0.690856, 0.496502);\n"
    "    vec3 c7 = vec3(0.319809, 0.770914, 0.411152);\n"
    "    vec3 c8 = vec3(0.525776, 0.833491, 0.288127);\n"
    "    vec3 c9 = vec3(0.762373, 0.876424, 0.137064);\n"
    "    vec3 c10 = vec3(0.993248, 0.906157, 0.143936);\n"
    "    if (t >= 1.0) return c10;\n"
    "    float s = t * 10.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    if (idx == 4) return mix(c4, c5, f);\n"
    "    if (idx == 5) return mix(c5, c6, f);\n"
    "    if (idx == 6) return mix(c6, c7, f);\n"
    "    if (idx == 7) return mix(c7, c8, f);\n"
    "    if (idx == 8) return mix(c8, c9, f);\n"
    "    return mix(c9, c10, f);\n"
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
    "    // Field synthesis: u(x,y,t) = Σ A_s / N_s * cos(kx*x) * cos(ky*y)\n"
    "    float u = 0.0;\n"
    "\n"
    "    for (int i = 0; i < 2048; i++) {\n"
    "        if (i >= iNumModes) break;\n"
    "\n"
    "        // Fetch mode data: (kx, ky, shell, inv_count)\n"
    "        float texCoord = (float(i) + 0.5) / float(iNumModes);\n"
    "        vec4 mode = texture(iModeData, texCoord);\n"
    "\n"
    "        float kx = mode.r;\n"
    "        float ky = mode.g;\n"
    "        int shell = int(mode.b);\n"
    "        float inv_count = mode.a;\n"
    "\n"
    "        // Get shell amplitude\n"
    "        float shellCoord = (float(shell) + 0.5) / float(iNumShells);\n"
    "        float A_s = texture(iShellAmps, shellCoord).r;\n"
    "\n"
    "        // Standing wave with x and y mirror symmetry\n"
    "        float mode_val;\n"
    "\n"
    "        if (iModeType == 0) {\n"
    "            // All modes: cos(kx*x) * cos(ky*y)\n"
    "            mode_val = cos(kx * p.x) * cos(ky * p.y);\n"
    "        } else if (iModeType == 1) {\n"
    "            // Only m != n modes (skip when kx == ky)\n"
    "            if (abs(kx - ky) < 0.01) continue;\n"
    "            mode_val = cos(kx * p.x) * cos(ky * p.y);\n"
    "        } else {\n"
    "            // Diagonal symmetry: cos(kx*x)*cos(ky*y) + cos(ky*x)*cos(kx*y)\n"
    "            mode_val = cos(kx * p.x) * cos(ky * p.y) + cos(ky * p.x) * cos(kx * p.y);\n"
    "            if (abs(kx - ky) < 0.01) mode_val *= 0.5;\n"
    "        }\n"
    "\n"
    "        u += A_s * inv_count * mode_val;\n"
    "    }\n"
    "\n"
    "    // Energy rendering: I = u^2\n"
    "    float I = u * u;\n"
    "\n"
    "    // Apply contrast\n"
    "    I = pow(I, 1.0 / iContrast);\n"
    "\n"
    "    // Soft clamp\n"
    "    I = tanh(I * 2.0);\n"
    "\n"
    "    // Color mapping\n"
    "    vec3 color;\n"
    "    if (iColorMode == 0) {\n"
    "        color = plasma(I);\n"
    "    } else if (iColorMode == 1) {\n"
    "        color = magma(I);\n"
    "    } else if (iColorMode == 2) {\n"
    "        color = turbo(I);\n"
    "    } else if (iColorMode == 3) {\n"
    "        color = viridis(I);\n"
    "    } else {\n"
    "        // Grayscale\n"
    "        color = vec3(I);\n"
    "    }\n"
    "\n"
    "    outColor = vec4(color, 1.0);\n"
    "}\n";

/* ============================================================================
 * Audio Capture (WASAPI Loopback)
 * ============================================================================ */

typedef struct {
    IAudioClient *pAudioClient;
    IAudioCaptureClient *pCaptureClient;
    WAVEFORMATEX *pwfx;
    HANDLE hThread;
    volatile BOOL running;
    volatile BOOL paused;

    float ringBuffer[MAX_FFT_SIZE * 4];
    volatile LONG writePos;

    int fftSize;
    float window[MAX_FFT_SIZE];
    float spectrum[MAX_BINS];
    float smoothSpectrum[MAX_BINS];
    float runningMax;
    float output[MAX_BINS];

    CRITICAL_SECTION cs;
} AudioCapture;

static AudioCapture g_audio;

/* Simple radix-2 FFT - Cooley-Tukey algorithm */
static void fft_complex(float *real, float *imag, int n) {
    int i, j, k;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = real[i], ti = imag[i];
            real[i] = real[j]; imag[i] = imag[j];
            real[j] = tr; imag[j] = ti;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * PI / len;
        float wpr = cosf(angle);
        float wpi = sinf(angle);
        for (i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (j = 0; j < len / 2; j++) {
                int u = i + j;
                int v = i + j + len / 2;
                float tr = wr * real[v] - wi * imag[v];
                float ti = wr * imag[v] + wi * real[v];
                real[v] = real[u] - tr;
                imag[v] = imag[u] - ti;
                real[u] += tr;
                imag[u] += ti;
                float wt = wr;
                wr = wr * wpr - wi * wpi;
                wi = wt * wpi + wi * wpr;
            }
        }
    }
}

static void AudioUpdateFFTParams(AudioCapture *a) {
    for (int i = 0; i < a->fftSize; i++) {
        a->window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (a->fftSize - 1)));
    }
    memset(a->spectrum, 0, sizeof(a->spectrum));
    memset(a->smoothSpectrum, 0, sizeof(a->smoothSpectrum));
    a->runningMax = 0.01f;
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

static BOOL AudioInit(AudioCapture *a, int fftSize) {
    HRESULT hr;
    IMMDeviceEnumerator *pEnumerator = NULL;
    IMMDevice *pDevice = NULL;

    memset(a, 0, sizeof(*a));
    a->fftSize = fftSize;
    a->runningMax = 0.01f;
    a->paused = FALSE;
    InitializeCriticalSection(&a->cs);
    AudioUpdateFFTParams(a);

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
    DeleteCriticalSection(&a->cs);
}

static void AudioSetFFTSize(AudioCapture *a, int size) {
    if (size < 256) size = 256;
    if (size > MAX_FFT_SIZE) size = MAX_FFT_SIZE;
    if (size != a->fftSize) {
        EnterCriticalSection(&a->cs);
        a->fftSize = size;
        AudioUpdateFFTParams(a);
        LeaveCriticalSection(&a->cs);
    }
}

static float *AudioGetData(AudioCapture *a) {
    static float fftReal[MAX_FFT_SIZE];
    static float fftImag[MAX_FFT_SIZE];

    int fftSize = a->fftSize;
    int nBins = fftSize / 2 + 1;
    int bufLen = MAX_FFT_SIZE * 4;

    EnterCriticalSection(&a->cs);
    int currentPos = a->writePos;
    int start = (currentPos - fftSize + bufLen) % bufLen;

    for (int i = 0; i < fftSize; i++) {
        int idx = (start + i) % bufLen;
        fftReal[i] = a->ringBuffer[idx] * a->window[i];
        fftImag[i] = 0.0f;
    }
    LeaveCriticalSection(&a->cs);

    fft_complex(fftReal, fftImag, fftSize);

    for (int i = 0; i < nBins; i++) {
        a->spectrum[i] = sqrtf(fftReal[i] * fftReal[i] + fftImag[i] * fftImag[i]);
    }

    float currentMax = 0.0f;
    for (int i = 0; i < nBins; i++) {
        if (a->spectrum[i] > currentMax) currentMax = a->spectrum[i];
    }

    if (currentMax > a->runningMax) {
        a->runningMax = currentMax;
    } else {
        a->runningMax = a->runningMax * 0.995f;
        if (a->runningMax < currentMax) a->runningMax = currentMax;
        if (a->runningMax < 0.01f) a->runningMax = 0.01f;
    }

    for (int i = 0; i < nBins; i++) {
        float normalized = a->spectrum[i] / (a->runningMax + 1e-6f);
        if (normalized > 1.5f) normalized = 1.5f;
        if (normalized < 0.0f) normalized = 0.0f;

        if (normalized > a->smoothSpectrum[i]) {
            a->smoothSpectrum[i] = normalized;
        } else {
            a->smoothSpectrum[i] = a->smoothSpectrum[i] * 0.85f + normalized * 0.15f;
        }
    }

    memset(a->output, 0, sizeof(a->output));
    for (int i = 0; i < nBins; i++) {
        a->output[i] = a->smoothSpectrum[i];
    }

    return a->output;
}

/* ============================================================================
 * Shell Processor - Constant-Q mode mapping
 * ============================================================================ */

typedef struct {
    int numModes;
    int numShells;
    float kappa;
    float tauAttack;
    float tauRelease;

    float modeData[MAX_MODES * 4];  /* (kx, ky, shell, inv_count) */
    int shellCounts[MAX_SHELLS];
    float shellAmps[MAX_SHELLS];

    LARGE_INTEGER lastTime;
    LARGE_INTEGER freq;
} ShellProcessor;

static ShellProcessor g_shell;

static int BinToShell(int binIdx, int nBins, float freqPerBin) {
    float freq = binIdx * freqPerBin;
    if (freq < 20.0f) return -1;

    float baseFreq = 40.0f;
    float r = freq / baseFreq;
    if (r < 1.0f) return -1;

    int s = (int)floorf(4.0f * log2f(r));
    return (s < MAX_SHELLS) ? s : MAX_SHELLS - 1;
}

static void ShellInit(ShellProcessor *sp) {
    memset(sp, 0, sizeof(*sp));
    sp->numShells = MAX_SHELLS;
    sp->kappa = 5.0f;
    sp->tauAttack = 0.040f;
    sp->tauRelease = 0.300f;

    QueryPerformanceFrequency(&sp->freq);
    QueryPerformanceCounter(&sp->lastTime);

    /* Generate modes */
    int modeIdx = 0;
    int maxMN = 32;

    for (int m = 0; m <= maxMN; m++) {
        for (int n = 0; n <= maxMN; n++) {
            if (m == 0 && n == 0) continue;  /* Skip DC */
            if (modeIdx >= MAX_MODES) break;

            float r = sqrtf((float)(m*m + n*n));
            int s = (int)floorf(4.0f * log2f(r));
            if (s < 0 || s >= MAX_SHELLS) continue;

            float kx = m * PI;
            float ky = n * PI;

            sp->modeData[modeIdx * 4 + 0] = kx;
            sp->modeData[modeIdx * 4 + 1] = ky;
            sp->modeData[modeIdx * 4 + 2] = (float)s;
            sp->modeData[modeIdx * 4 + 3] = 0.0f;  /* Will set inv_count later */
            sp->shellCounts[s]++;
            modeIdx++;
        }
        if (modeIdx >= MAX_MODES) break;
    }
    sp->numModes = modeIdx;

    /* Set inverse counts */
    for (int i = 0; i < sp->numModes; i++) {
        int s = (int)sp->modeData[i * 4 + 2];
        float invCount = 1.0f / (float)(sp->shellCounts[s] > 0 ? sp->shellCounts[s] : 1);
        sp->modeData[i * 4 + 3] = invCount;
    }

    int activeShells = 0;
    for (int i = 0; i < MAX_SHELLS; i++) {
        if (sp->shellCounts[i] > 0) activeShells++;
    }

    printf("Initialized %d modes across %d active shells\n", sp->numModes, activeShells);
}

static void ShellProcess(ShellProcessor *sp, float *spectrum, int nBins, int fftSize) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    float dt = (float)(now.QuadPart - sp->lastTime.QuadPart) / (float)sp->freq.QuadPart;
    sp->lastTime = now;

    float freqPerBin = (float)SAMPLE_RATE / fftSize;

    /* Compute raw shell energies */
    float shellEnergyRaw[MAX_SHELLS] = {0};
    for (int i = 1; i < nBins; i++) {
        int s = BinToShell(i, nBins, freqPerBin);
        if (s >= 0 && s < MAX_SHELLS) {
            shellEnergyRaw[s] += spectrum[i] * spectrum[i];
        }
    }

    /* Log compression and attack/release smoothing */
    float alphaAttack = 1.0f - expf(-dt / sp->tauAttack);
    float alphaRelease = 1.0f - expf(-dt / sp->tauRelease);

    for (int s = 0; s < MAX_SHELLS; s++) {
        float E_s = logf(1.0f + sp->kappa * shellEnergyRaw[s]);

        if (E_s > sp->shellAmps[s]) {
            sp->shellAmps[s] += alphaAttack * (E_s - sp->shellAmps[s]);
        } else {
            sp->shellAmps[s] += alphaRelease * (E_s - sp->shellAmps[s]);
        }
    }
}

/* ============================================================================
 * Visualizer
 * ============================================================================ */

#define DEFAULT_CONTRAST     1.5f
#define DEFAULT_SCALE        1.0f
#define DEFAULT_COLOR_MODE   0
#define DEFAULT_MODE_TYPE    0
#define DEFAULT_FFT_SIZE     8192

#define NUM_COLOR_MODES      5
#define NUM_MODE_TYPES       3

static const char *COLOR_NAMES[] = {"Plasma", "Magma", "Turbo", "Viridis", "Grayscale"};
static const char *MODE_TYPE_NAMES[] = {"All", "m!=n", "Diagonal"};

typedef struct {
    int w, h;
    float contrast;
    float scale;
    int colorMode;
    int modeType;
    float time;

    GLuint program;
    GLuint vao;
    GLuint shellTex;
    GLuint modeTex;

    GLint locResolution;
    GLint locShellAmps;
    GLint locModeData;
    GLint locNumModes;
    GLint locNumShells;
    GLint locTime;
    GLint locColorMode;
    GLint locContrast;
    GLint locScale;
    GLint locModeType;
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
    v->contrast = DEFAULT_CONTRAST;
    v->scale = DEFAULT_SCALE;
    v->colorMode = DEFAULT_COLOR_MODE;
    v->modeType = DEFAULT_MODE_TYPE;
}

static BOOL VizInit(Visualizer *v, int w, int h, ShellProcessor *sp) {
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

    /* 1D texture for shell amplitudes */
    glGenTextures(1, &v->shellTex);
    glBindTexture(GL_TEXTURE_1D, v->shellTex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, MAX_SHELLS, 0, GL_RED, GL_FLOAT, NULL);

    /* 1D texture for mode data */
    glGenTextures(1, &v->modeTex);
    glBindTexture(GL_TEXTURE_1D, v->modeTex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, sp->numModes, 0, GL_RGBA, GL_FLOAT, sp->modeData);

    v->locResolution = glGetUniformLocation(v->program, "iResolution");
    v->locShellAmps = glGetUniformLocation(v->program, "iShellAmps");
    v->locModeData = glGetUniformLocation(v->program, "iModeData");
    v->locNumModes = glGetUniformLocation(v->program, "iNumModes");
    v->locNumShells = glGetUniformLocation(v->program, "iNumShells");
    v->locTime = glGetUniformLocation(v->program, "iTime");
    v->locColorMode = glGetUniformLocation(v->program, "iColorMode");
    v->locContrast = glGetUniformLocation(v->program, "iContrast");
    v->locScale = glGetUniformLocation(v->program, "iScale");
    v->locModeType = glGetUniformLocation(v->program, "iModeType");

    return TRUE;
}

static void VizRender(Visualizer *v, ShellProcessor *sp, float dt) {
    v->time += dt;

    /* Update shell amplitudes texture */
    glBindTexture(GL_TEXTURE_1D, v->shellTex);
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, MAX_SHELLS, GL_RED, GL_FLOAT, sp->shellAmps);

    glViewport(0, 0, v->w, v->h);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(v->program);
    glUniform2f(v->locResolution, (float)v->w, (float)v->h);
    glUniform1i(v->locNumModes, sp->numModes);
    glUniform1i(v->locNumShells, MAX_SHELLS);
    glUniform1f(v->locTime, v->time);
    glUniform1i(v->locColorMode, v->colorMode);
    glUniform1f(v->locContrast, v->contrast);
    glUniform1f(v->locScale, v->scale);
    glUniform1i(v->locModeType, v->modeType);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, v->shellTex);
    glUniform1i(v->locShellAmps, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, v->modeTex);
    glUniform1i(v->locModeData, 1);

    glBindVertexArray(v->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);
}

static void PrintStatus(Visualizer *v, ShellProcessor *sp, int fftSize) {
    char status[256];
    snprintf(status, sizeof(status),
             "Scale=%.2f  Contrast=%.1f  Kappa=%.1f  Attack=%.0fms  Release=%.0fms  Mode=%s  Color=%s  FFT=%d",
             v->scale, v->contrast, sp->kappa,
             sp->tauAttack * 1000.0f, sp->tauRelease * 1000.0f,
             MODE_TYPE_NAMES[v->modeType], COLOR_NAMES[v->colorMode], fftSize);
    printf("\r%-140s", status);
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

    g_hwnd = CreateWindow("ShellsClass", "Spectral Shell Visualizer", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
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

    ShellInit(&g_shell);

    if (!VizInit(&g_viz, winW, winH, &g_shell)) {
        printf("Failed to initialize visualizer\n");
        DestroyOpenGLContext();
        return 1;
    }

    if (!AudioInit(&g_audio, DEFAULT_FFT_SIZE)) {
        printf("Failed to initialize audio capture\n");
        DestroyOpenGLContext();
        return 1;
    }

    Sleep(50);

    printf("Spectral Shell Visualizer - Constant-Q Standing Wave Synthesis\n");
    printf("Controls: W/S=scale  Z/X=contrast  K/L=kappa  UP/DOWN=attack  LEFT/RIGHT=release\n");
    printf("          M=mode type  V=color  F=FFT size  ALT+ENTER=fullscreen  SPACE=reset  ESC=quit\n\n");

    PrintStatus(&g_viz, &g_shell, g_audio.fftSize);

    BOOL running = TRUE;
    DWORD lastKeyTime = 0;
    DWORD repeatDelay = 100;
    BOOL prevF = FALSE, prevM = FALSE, prevV = FALSE, prevSpace = FALSE;
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

            /* Contrast Z/X */
            if ((GetAsyncKeyState('X') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.contrast += 0.1f;
                if (g_viz.contrast > 5.0f) g_viz.contrast = 5.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('Z') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.contrast -= 0.1f;
                if (g_viz.contrast < 0.1f) g_viz.contrast = 0.1f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Kappa K/L */
            if ((GetAsyncKeyState('L') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_shell.kappa *= 1.2f;
                if (g_shell.kappa > 50.0f) g_shell.kappa = 50.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('K') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_shell.kappa /= 1.2f;
                if (g_shell.kappa < 0.1f) g_shell.kappa = 0.1f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Attack UP/DOWN */
            if ((GetAsyncKeyState(VK_UP) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_shell.tauAttack *= 1.2f;
                if (g_shell.tauAttack > 0.5f) g_shell.tauAttack = 0.5f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_DOWN) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_shell.tauAttack /= 1.2f;
                if (g_shell.tauAttack < 0.001f) g_shell.tauAttack = 0.001f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Release LEFT/RIGHT */
            if ((GetAsyncKeyState(VK_RIGHT) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_shell.tauRelease *= 1.2f;
                if (g_shell.tauRelease > 2.0f) g_shell.tauRelease = 2.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_LEFT) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_shell.tauRelease /= 1.2f;
                if (g_shell.tauRelease < 0.01f) g_shell.tauRelease = 0.01f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* FFT size F (edge-triggered) */
            BOOL currF = (GetAsyncKeyState('F') & 0x8000) != 0;
            if (currF && !prevF) {
                int newSize = g_audio.fftSize * 2;
                if (newSize > MAX_FFT_SIZE) newSize = 256;
                AudioSetFFTSize(&g_audio, newSize);
                needUpdate = TRUE;
            }
            prevF = currF;

            /* Mode type M (edge-triggered) */
            BOOL currM = (GetAsyncKeyState('M') & 0x8000) != 0;
            if (currM && !prevM) {
                g_viz.modeType = (g_viz.modeType + 1) % NUM_MODE_TYPES;
                needUpdate = TRUE;
            }
            prevM = currM;

            /* Color mode V (edge-triggered) */
            BOOL currV = (GetAsyncKeyState('V') & 0x8000) != 0;
            if (currV && !prevV) {
                g_viz.colorMode = (g_viz.colorMode + 1) % NUM_COLOR_MODES;
                needUpdate = TRUE;
            }
            prevV = currV;

            /* Reset SPACE (edge-triggered) */
            BOOL currSpace = (GetAsyncKeyState(VK_SPACE) & 0x8000) != 0;
            if (currSpace && !prevSpace) {
                VizResetDefaults(&g_viz);
                g_shell.kappa = 5.0f;
                g_shell.tauAttack = 0.040f;
                g_shell.tauRelease = 0.300f;
                AudioSetFFTSize(&g_audio, DEFAULT_FFT_SIZE);
                needUpdate = TRUE;
            }
            prevSpace = currSpace;
        } else {
            prevF = FALSE;
            prevM = FALSE;
            prevV = FALSE;
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
            float *spectrum = AudioGetData(&g_audio);
            if (spectrum) {
                int nBins = g_audio.fftSize / 2 + 1;
                ShellProcess(&g_shell, spectrum, nBins, g_audio.fftSize);
                VizRender(&g_viz, &g_shell, dt);
                SwapBuffers(g_hdc);

                if (needUpdate) {
                    PrintStatus(&g_viz, &g_shell, g_audio.fftSize);
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

