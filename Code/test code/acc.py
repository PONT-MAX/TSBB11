## Step #1. Obtain an OpenCL platform. Comupter
platform = cl.get_platforms()[0]
## Step #2. Obtain a device id for at least one device (accelerator). Grapics cors
device = platform.get_devices()[1]
## Step #3. Create a context for the selected device. form 1 & 2
ctx = cl.Context([device])

## Step #4. Create the accelerator program from source code.
kernel_code = """
    __kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g){
        int gid = get_global_id(0);
        res_g[gid] = a_g[gid] + b_g[gid];
    }

    __kernel void sub(__global const float *a_g, __global const float *b_g, __global float *res_g){
        int gid = get_global_id(0);
        res_g[gid] = 5 * a_g[gid] - b_g[gid];
    }
    """

## Step #5. Build the program.
## Step #6. Create one or more kernels from the program functions.
prg = cl.Program(ctx,kernel_code).build()

## Step #7. Create a command queue for the target device.
queue = cl.CommandQueue(ctx)

## Step #8. Allocate device memory and move input data from the host to the device memory.
mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
res_np = np.empty_like(a_np)

## Step #9. Associate the arguments to the kernel with kernel object.
## Step #10. Deploy the kernel for device execution.
prg.sub(queue, a_np.shape, None, a_g, b_g, res_g)

## Step #11. Move the kernels output data to host memory.
cl.enqueue_copy(queue, res_np, res_g)

## Step #12. Release context, program, kernels and memory.
## PyOpenCL performs this step for you, and therefore,
## you don't need to worry about cleanup code






print("Heeej")
# Check on CPU with Numpy:
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))
#print(c_np)

"""
# load the image and convert it to grayscale
image = cv2.imread("../Data/ortho/0152945e_582034n_20160905T073402Z_tex.tif")
imsec = image[2000:3000,2000:3000]
cv2.imshow("Original", imsec)

cv2.waitKey(0)
"""