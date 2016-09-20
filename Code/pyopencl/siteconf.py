CL_TRACE = False
CL_ENABLE_GL = False
CL_INC_DIR = []
CL_LIB_DIR = []
CL_LIBNAME = []
CXXFLAGS = ['-std=c++0x', '-stdlib=libc++', '-mmacosx-version-min=10.7', '-arch', 'i386', '-arch', 'x86_64']
LDFLAGS = ['-std=c++0x', '-stdlib=libc++', '-mmacosx-version-min=10.7', '-arch', 'i386', '-arch', 'x86_64', '-Wl,-framework,OpenCL']
