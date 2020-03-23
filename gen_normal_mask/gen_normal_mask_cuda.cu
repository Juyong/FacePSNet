#include "cuda_config.h"

template <typename Dtype>
__device__ bool is_in_tri(Dtype *vpos, Dtype *v1pos, Dtype *v2pos, Dtype *v3pos,
                          Dtype *tricoord) {
  Dtype x1 = v2pos[0] - v1pos[0];
  Dtype y1 = v2pos[1] - v1pos[1];

  Dtype x2 = v3pos[0] - v1pos[0];
  Dtype y2 = v3pos[1] - v1pos[1];

  Dtype x = vpos[0] - v1pos[0];
  Dtype y = vpos[1] - v1pos[1];

  if (x1 * y2 == x2 * y1) {
    return false;
  }

  Dtype b = (x * y1 - x1 * y) / (x2 * y1 - x1 * y2);
  Dtype a = (x * y2 - x2 * y) / (x1 * y2 - x2 * y1);

  tricoord[0] = Dtype(1) - a - b;
  tricoord[1] = a;
  tricoord[2] = b;

  float eps = 0;

  if (a >= 0-eps && b >= 0-eps && (a + b) <= 1+eps) {
    return true;
  }

  return false;
}


__global__ void gen_normal_mask_kernel(float *proj_geo_, float *tri_normal_, long *tri_inds, float *depthBuffer_,
    int *dBufferLocked_, float *normal_map_, float *mask_, int batch_size,
    int point_num, int tri_num, int height, int width)
{
    CUDA_KERNEL_LOOP(index, tri_num * batch_size)
    {
        int n = index / tri_num;
        int tri_id = index % tri_num;

        int pos = height * width;
        float *depthBuffer = depthBuffer_ + n * pos;
        float *normal_map = normal_map_ + n*3*pos;
        float* mask = mask_ + n*pos;
        float* tri_normal = tri_normal_ + 3*n*tri_num;
        float *proj_geo = proj_geo_ + n * 3 * point_num;
        int *dBufferLocked = dBufferLocked_ + n * pos;

        int ind1 = tri_inds[3 * tri_id], ind2 = tri_inds[3 * tri_id + 1],
            ind3 = tri_inds[3 * tri_id + 2];

        float vpos9[9];
        float *v1imgpos = vpos9, *v2imgpos = vpos9 + 3, *v3imgpos = vpos9 + 6;
        for (int j = 0; j < 3; j++) {
            v1imgpos[j] = proj_geo[3 * ind1 + j];
            v2imgpos[j] = proj_geo[3 * ind2 + j];
            v3imgpos[j] = proj_geo[3 * ind3 + j];
        }

        float xmin = width, xmax = -2, ymin = height, ymax = -2;
        xmin = fminf(xmin, v1imgpos[0]);
        ymin = fminf(ymin, v1imgpos[1]);
        xmin = fminf(xmin, v2imgpos[0]);
        ymin = fminf(ymin, v2imgpos[1]);
        xmin = fminf(xmin, v3imgpos[0]);
        ymin = fminf(ymin, v3imgpos[1]);

        xmax = fmaxf(xmax, v1imgpos[0]);
        ymax = fmaxf(ymax, v1imgpos[1]);
        xmax = fmaxf(xmax, v2imgpos[0]);
        ymax = fmaxf(ymax, v2imgpos[1]);
        xmax = fmaxf(xmax, v3imgpos[0]);
        ymax = fmaxf(ymax, v3imgpos[1]);

        xmin = fmaxf(float(0), xmin);
        ymin = fmaxf(float(0), ymin);
        xmax = fminf(xmax, float(width - 2));
        ymax = fminf(ymax, float(height - 2));

        float coord[3], vpos[2];
        for (int x = ceilf(xmin); x <= floorf(xmax); x++) {
            for (int y = ceilf(ymin); y <= floorf(ymax); y++) {
                vpos[0] = x;
                vpos[1] = y;
                if (is_in_tri(vpos, v1imgpos, v2imgpos, v3imgpos, coord)) {
                    int pixelindex = y * width + x;
                    float z_value = v1imgpos[2]*coord[0] + v2imgpos[2]*coord[1] + v3imgpos[2]*coord[2];
                    bool wait = true;
                    while (wait) {
                        if (0 == atomicExch(&dBufferLocked[pixelindex], 1)) {
                            if (z_value > depthBuffer[pixelindex]) {
                                depthBuffer[pixelindex] = z_value;
                                mask[pixelindex] = 1;
                                for(int j=0; j<3; j++)
                                {
                                    if(tri_normal[3*tri_id+2]>0.001)
                                    {
                                        normal_map[j*pos+pixelindex] = tri_normal[3*tri_id+j];
                                    }
                                }
                                // normal_map[0*pos+pixelindex] = 0;
                                // normal_map[1*pos+pixelindex] = 0;
                                // normal_map[2*pos+pixelindex] = 1;
                            }
                            wait = false;
                            dBufferLocked[pixelindex] = 0;
                        }
                    }
                }
            }
        }
    }
}

__global__ void gen_depth_kernel(float *proj_geo_, long *tri_inds, float *depthBuffer_,
    int *dBufferLocked_, int batch_size,
    int point_num, int tri_num, int height, int width)
{
    CUDA_KERNEL_LOOP(index, tri_num * batch_size)
    {
        int n = index / tri_num;
        int tri_id = index % tri_num;

        int pos = height * width;
        float *depthBuffer = depthBuffer_ + n * pos;
        float *proj_geo = proj_geo_ + n * 3 * point_num;
        int *dBufferLocked = dBufferLocked_ + n * pos;

        int ind1 = tri_inds[3 * tri_id], ind2 = tri_inds[3 * tri_id + 1],
            ind3 = tri_inds[3 * tri_id + 2];

        float vpos9[9];
        float *v1imgpos = vpos9, *v2imgpos = vpos9 + 3, *v3imgpos = vpos9 + 6;
        for (int j = 0; j < 3; j++) {
            v1imgpos[j] = proj_geo[3 * ind1 + j];
            v2imgpos[j] = proj_geo[3 * ind2 + j];
            v3imgpos[j] = proj_geo[3 * ind3 + j];
        }

        float xmin = width, xmax = -2, ymin = height, ymax = -2;
        xmin = fminf(xmin, v1imgpos[0]);
        ymin = fminf(ymin, v1imgpos[1]);
        xmin = fminf(xmin, v2imgpos[0]);
        ymin = fminf(ymin, v2imgpos[1]);
        xmin = fminf(xmin, v3imgpos[0]);
        ymin = fminf(ymin, v3imgpos[1]);

        xmax = fmaxf(xmax, v1imgpos[0]);
        ymax = fmaxf(ymax, v1imgpos[1]);
        xmax = fmaxf(xmax, v2imgpos[0]);
        ymax = fmaxf(ymax, v2imgpos[1]);
        xmax = fmaxf(xmax, v3imgpos[0]);
        ymax = fmaxf(ymax, v3imgpos[1]);

        xmin = fmaxf(float(0), xmin);
        ymin = fmaxf(float(0), ymin);
        xmax = fminf(xmax, float(width - 2));
        ymax = fminf(ymax, float(height - 2));

        float coord[3], vpos[2];
        for (int x = ceilf(xmin); x <= floorf(xmax); x++) {
            for (int y = ceilf(ymin); y <= floorf(ymax); y++) {
                vpos[0] = x;
                vpos[1] = y;
                if (is_in_tri(vpos, v1imgpos, v2imgpos, v3imgpos, coord)) {
                    int pixelindex = y * width + x;
                    float z_value = v1imgpos[2]*coord[0] + v2imgpos[2]*coord[1] + v3imgpos[2]*coord[2];
                    bool wait = true;
                    while (wait) {
                        if (0 == atomicExch(&dBufferLocked[pixelindex], 1)) {
                            if (z_value > depthBuffer[pixelindex]) {
                                depthBuffer[pixelindex] = z_value;
                            }
                            wait = false;
                            dBufferLocked[pixelindex] = 0;
                        }
                    }
                }
            }
        }
    }
}


void gen_normal_mask_cuda(float *proj_geo, float *tri_normal, long *tri_inds, float *depthBuffer,
    int *dBufferLocked, float *normal_map, float *mask, int batch_size,
    int point_num, int tri_num, int height, int width)
{
    gen_normal_mask_kernel<<<GET_BLOCKS(batch_size * tri_num), CUDA_NUM_THREADS>>>
        (proj_geo, tri_normal, tri_inds, depthBuffer, dBufferLocked, 
        normal_map, mask, batch_size,point_num, tri_num, height, width);
}


void gen_depth_cuda(float *proj_geo, long *tri_inds, float *depthBuffer,
    int *dBufferLocked, int batch_size,
    int point_num, int tri_num, int height, int width)
{
    gen_depth_kernel<<<GET_BLOCKS(batch_size * tri_num), CUDA_NUM_THREADS>>>
        (proj_geo, tri_inds, depthBuffer, dBufferLocked, 
        batch_size,point_num, tri_num, height, width);
}