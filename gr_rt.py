import cupy as cp
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import os

OUT_DIR    = "bh_frames"
VIDEO_FILE = "binary_bh.mp4"
os.makedirs(OUT_DIR, exist_ok=True)

# Video parameters
IMG_W      = 960
IMG_H      = 540
NUM_FRAMES = 360      # frames total (3 full orbits at 24fps = 15s)
FPS        = 24
SPP        = 4        # anti-alias samples. 1=fast, 8=nice

# Black hole and camera parameters
BH_SEP     = 14.0     # BH1 at +x, BH2 at -x — farther apart
RS         = 1.0      # Schwarzschild radius per BH
DISK_INNER = 3.5      # inner accretion disk radius
DISK_OUTER = 16.0     # outer accretion disk radius
CAM_DIST   = 55.0     # camera distance from origin
CAM_Y      = 1.5      # camera height above disk plane (keep small for edge-on view)

# Cuda Kerner
KERNEL_SRC = r"""

__device__ __forceinline__ float3 f3(float x,float y,float z){return make_float3(x,y,z);}
__device__ __forceinline__ float  dot3(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ __forceinline__ float  len3(float3 v){return sqrtf(dot3(v,v));}
__device__ __forceinline__ float3 add3(float3 a,float3 b){return f3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ __forceinline__ float3 sub3(float3 a,float3 b){return f3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ __forceinline__ float3 mul3(float3 v,float s)  {return f3(v.x*s,v.y*s,v.z*s);}
__device__ __forceinline__ float3 nrm3(float3 v){return mul3(v,1.f/(len3(v)+1e-12f));}
__device__ __forceinline__ float3 cross3(float3 a,float3 b){
    return f3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);}

/* ---------- GR geodesic acceleration ---------- */
__device__ float3 accel(float3 p, float3 bh1, float3 bh2, float rs)
{
    float3 r1=sub3(p,bh1); float d1=len3(r1)+1e-8f;
    float3 r2=sub3(p,bh2); float d2=len3(r2)+1e-8f;
    float f1 = -1.5f*rs/(d1*d1*d1);
    float f2 = -1.5f*rs/(d2*d2*d2);
    return f3(r1.x*f1+r2.x*f2,
              r1.y*f1+r2.y*f2,
              r1.z*f1+r2.z*f2);
}

/* ---------- disk emission ---------- */
__device__ bool disk_hit(float3 p0, float3 p1,
                          float inner, float outer,
                          float3 bh1, float3 bh2, float rs,
                          float &er, float &eg, float &eb, float &alpha)
{
    /* must cross y=0 */
    if ((p0.y >= 0.f) == (p1.y >= 0.f)) return false;
    float t  = p0.y / (p0.y - p1.y + 1e-12f);
    float cx = p0.x + t*(p1.x-p0.x);
    float cz = p0.z + t*(p1.z-p0.z);
    float r  = sqrtf(cx*cx + cz*cz);
    if (r < inner || r > outer) return false;

    /* inside BH shadow radius -> invisible */
    float dx1=cx-bh1.x, dz1=cz-bh1.z;
    float dx2=cx-bh2.x, dz2=cz-bh2.z;
    if (sqrtf(dx1*dx1+dz1*dz1) < rs*1.1f) return false;
    if (sqrtf(dx2*dx2+dz2*dz2) < rs*1.1f) return false;

    float norm_r = (r - inner) / (outer - inner + 1e-6f);  /* 0=inner,1=outer */
    float inv_r  = 1.f - norm_r;

    /* Temperature */
    float T = powf(inv_r, 2.2f);

    /* Doppler brightening */
    float phi     = atan2f(cz, cx);
    float doppler = 0.55f + 0.45f*sinf(phi);

    /* Gravitational redshift */
    float d1 = sqrtf(dx1*dx1+dz1*dz1)+1e-6f;
    float d2 = sqrtf(dx2*dx2+dz2*dz2)+1e-6f;
    float gshift = 1.f - 0.5f*(rs/fmaxf(d1,rs+0.01f) + rs/fmaxf(d2,rs+0.01f));
    gshift = fmaxf(0.05f, gshift);

    float em = T * doppler * gshift * 2.5f;

    /* Blackbody colour */
    er    = fminf(1.f, em * 1.0f + 0.05f);
    eg    = fminf(1.f, em * 0.55f - 0.05f);
    eb    = fminf(1.f, em * 0.15f - 0.05f);
    er    = fmaxf(0.f, er);
    eg    = fmaxf(0.f, eg);
    eb    = fmaxf(0.f, eb);
    alpha = fminf(1.f, inv_r * 1.2f + 0.05f);
    return true;
}

/* ---------- hash starfield ---------- */
__device__ float star_lum(float3 d)
{
    int ix=(int)((d.x+2.f)*4000.f);
    int iy=(int)((d.y+2.f)*4000.f);
    int iz=(int)((d.z+2.f)*4000.f);
    unsigned int h=(unsigned int)(ix*2654435761u ^ iy*2246822519u ^ iz*3266489917u);
    float v=(float)(h&0xFFFF)/65535.f;
    return (v>0.9978f) ? powf((v-0.9978f)/0.0022f,1.5f) : 0.f;
}

/* ---------- main kernel ---------- */
extern "C" __global__
void raytrace(
    float* __restrict__ out,
    int W, int H,
    /* camera */
    float cox, float coy, float coz,
    float fdx, float fdy, float fdz,
    float rdx, float rdy, float rdz,
    float udx, float udy, float udz,
    float fov_tan, float aspect,
    /* scene */
    float bh1x, float bh1z,
    float bh2x, float bh2z,
    float rs, float d_inner, float d_outer,
    /* aa */
    int spp, unsigned long long seed)
{
    int px = blockIdx.x*blockDim.x + threadIdx.x;
    int py = blockIdx.y*blockDim.y + threadIdx.y;
    if (px>=W || py>=H) return;

    float3 cam_o = f3(cox,coy,coz);
    float3 fwd   = f3(fdx,fdy,fdz);
    float3 right = f3(rdx,rdy,rdz);
    float3 up    = f3(udx,udy,udz);
    float3 bh1   = f3(bh1x, 0.f, bh1z);
    float3 bh2   = f3(bh2x, 0.f, bh2z);

    float acc_r=0.f, acc_g=0.f, acc_b=0.f;

    /* per-pixel LCG seed — no GCC extensions, plain arithmetic */
    unsigned long long rng = (unsigned long long)(py*W+px)*2654435761ULL ^ seed;
    rng = rng*6364136223846793005ULL + 1442695040888963407ULL;

    for (int s=0; s<spp; s++) {
        /* jitter for anti-aliasing */
        rng = rng*6364136223846793005ULL + 1442695040888963407ULL;
        float jx = (spp>1) ? (float)((rng>>33)&0x7FFFFFFF)/2147483648.f : 0.5f;
        rng = rng*6364136223846793005ULL + 1442695040888963407ULL;
        float jy = (spp>1) ? (float)((rng>>33)&0x7FFFFFFF)/2147483648.f : 0.5f;

        float ndcx = ((px+jx)/W*2.f - 1.f) * aspect * fov_tan;
        float ndcy = ((py+jy)/H*2.f - 1.f) * fov_tan;   /* +y = up */

        float3 dir = nrm3(add3(add3(fwd, mul3(right,ndcx)), mul3(up,ndcy)));

        float3 pos = cam_o;
        float3 vel = dir;
        float  cr=0.f, cg=0.f, cb=0.f, tr=1.f;
        bool   hit_bh = false;

        /* adaptive step size: finer when close to a BH */
        for (int i=0; i<1000; i++) {
            float d1 = len3(sub3(pos,bh1));
            float d2 = len3(sub3(pos,bh2));
            float dmin = fminf(d1,d2);
            float h = fminf(0.08f, fmaxf(0.015f, dmin*0.035f));

            float3 prev = pos;

            /* RK4 geodesic integration */
            float3 a1 = accel(pos, bh1, bh2, rs);
            float3 pm = add3(pos, mul3(vel, h*0.5f));
            float3 vm = add3(vel, mul3(a1,  h*0.5f));
            float3 a2 = accel(pm, bh1, bh2, rs);
            float3 pm2= add3(pos, mul3(vm, h*0.5f));
            float3 vm2= add3(vel, mul3(a2,  h*0.5f));
            float3 a3 = accel(pm2,bh1, bh2, rs);
            float3 pe = add3(pos, mul3(vm2,h));
            float3 ve = add3(vel, mul3(a3,  h));
            float3 a4 = accel(pe, bh1, bh2, rs);

            pos = add3(pos, mul3(
                add3(add3(add3(vel, mul3(vm,2.f)), mul3(vm2,2.f)), ve),
                h/6.f));
            vel = nrm3(add3(vel, mul3(
                add3(add3(add3(a1,  mul3(a2, 2.f)), mul3(a3, 2.f)), a4),
                h/6.f)));

            /* event horizon */
            float nd1=len3(sub3(pos,bh1)), nd2=len3(sub3(pos,bh2));
            if (nd1 < rs*0.9f || nd2 < rs*0.9f) { hit_bh=true; break; }

            /* disk */
            float er,eg,eb,al;
            if (disk_hit(prev, pos, d_inner, d_outer, bh1, bh2, rs, er,eg,eb,al)) {
                cr += tr*al*er;
                cg += tr*al*eg;
                cb += tr*al*eb;
                tr *= (1.f - al*0.85f);
                if (tr < 0.01f) break;
            }

            /* escape */
            if (len3(pos) > 150.f) break;
        }

        /* background stars */
        if (!hit_bh && tr > 0.02f) {
            float sv = star_lum(vel);
            cr += tr*sv; cg += tr*sv; cb += tr*sv*0.85f;
        }

        acc_r+=cr; acc_g+=cg; acc_b+=cb;
    }

    float inv=1.f/spp;
    float r=acc_r*inv*1.6f, g=acc_g*inv*1.6f, b=acc_b*inv*1.6f;

    /* ACES tone map */
    r = r*(2.51f*r+0.03f)/(r*(2.43f*r+0.59f)+0.14f);
    g = g*(2.51f*g+0.03f)/(g*(2.43f*g+0.59f)+0.14f);
    b = b*(2.51f*b+0.03f)/(b*(2.43f*b+0.59f)+0.14f);

    r=fminf(1.f,fmaxf(0.f,r));
    g=fminf(1.f,fmaxf(0.f,g));
    b=fminf(1.f,fmaxf(0.f,b));

    /* gamma 2.2 */
    r=powf(r,0.4545f); g=powf(g,0.4545f); b=powf(b,0.4545f);

    int idx=(py*W+px)*3;
    out[idx]=r; out[idx+1]=g; out[idx+2]=b;
}
"""

# Compile CUDA Kernel
print("Compiling CUDA kernel.", flush=True)
mod    = cp.RawModule(code=KERNEL_SRC)
kernel = mod.get_function("raytrace")
print("OK\n", flush=True)

# Render
buf   = cp.zeros(IMG_H * IMG_W * 3, dtype=cp.float32)
block = (16, 16, 1)
grid  = ((IMG_W+15)//16, (IMG_H+15)//16, 1)

bh1_pos = np.array([ BH_SEP/2, 0, 0], dtype=np.float32)
bh2_pos = np.array([-BH_SEP/2, 0, 0], dtype=np.float32)

frame_paths = []
print(f"Rendering {NUM_FRAMES} frames at {IMG_W}x{IMG_H}, SPP={SPP}")

for fi in tqdm(range(NUM_FRAMES)):
    # Camera orbits in the XZ plane, slightly above disk
    angle    = 6*np.pi * fi / NUM_FRAMES   # 3 full orbits
    cam_x    = CAM_DIST * np.sin(angle)
    cam_z    = CAM_DIST * np.cos(angle)
    cam_pos  = np.array([cam_x, CAM_Y, cam_z], dtype=np.float64)

    # Camera basis
    fwd   = -cam_pos / np.linalg.norm(cam_pos)
    wup   = np.array([0., 1., 0.])
    right = np.cross(fwd, wup); right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)           # true camera up (slightly tilted)

    fov_tan = float(np.tan(np.radians(25)))
    aspect  = IMG_W / IMG_H

    buf[:] = 0
    kernel(grid, block, (
        buf,
        np.int32(IMG_W), np.int32(IMG_H),
        # camera
        np.float32(cam_pos[0]), np.float32(cam_pos[1]), np.float32(cam_pos[2]),
        np.float32(fwd[0]),   np.float32(fwd[1]),   np.float32(fwd[2]),
        np.float32(right[0]), np.float32(right[1]), np.float32(right[2]),
        np.float32(up[0]),    np.float32(up[1]),    np.float32(up[2]),
        np.float32(fov_tan),  np.float32(aspect),
        # scene
        np.float32(bh1_pos[0]), np.float32(bh1_pos[2]),
        np.float32(bh2_pos[0]), np.float32(bh2_pos[2]),
        np.float32(RS),
        np.float32(DISK_INNER), np.float32(DISK_OUTER),
        # aa
        np.int32(SPP),
        np.uint64(fi * 1_000_003 + 42),
    ))
    cp.cuda.stream.get_current_stream().synchronize()

    img = (cp.asnumpy(buf).reshape(IMG_H, IMG_W, 3) * 255).clip(0, 255).astype(np.uint8)
    path = os.path.join(OUT_DIR, f"frame_{fi:04d}.png")
    imageio.imwrite(path, img)
    frame_paths.append(path)

# Encoding
print(f"\nEncoding {VIDEO_FILE}.")
with imageio.get_writer(VIDEO_FILE, fps=FPS, codec="libx264",
                        output_params=["-crf","17","-pix_fmt","yuv420p"]) as w:
    for p in tqdm(frame_paths):
        w.append_data(imageio.imread(p))
