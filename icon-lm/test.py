import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_USE_CUDNN'] = '1'

# The following are the key variables to prevent double registration
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDA_CLANG'] = '1'
import sys

# Set environment variables for detailed logging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['JAX_LOG_LEVEL'] = '1'

# Print general environment variables
print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))

# Check TensorFlow (as before)
try:
    import tensorflow as tf
    print("\nTensorFlow Version:", tf.__version__)
    print("TensorFlow Devices:", tf.config.list_physical_devices())
    print("TensorFlow CUDA Build Info:")
    print(tf.sysconfig.get_build_info())
except Exception as e:
    print("TensorFlow Initialization Error:", e)
                                                 

# Check JAX
try:
    import jax
    print("\nJAX Version:", jax.__version__)
    print("JAX Devices:", jax.devices())
    
    # Print JAX-specific environment variables
    print("JAX_PLATFORM_NAME:", os.environ.get('JAX_PLATFORM_NAME'))
    print("XLA_FLAGS:", os.environ.get('XLA_FLAGS'))
    
    # Check CUDA paths used by JAX
    from jax.lib import xla_bridge
    print("JAX CUDA Path:", xla_bridge.get_backend().platform_version)
except Exception as e:
    print("JAX Initialization Error:", e)

# Check PyTorch
try:
    import torch
    print("\nPyTorch Version:", torch.__version__)
    print("PyTorch CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("PyTorch CUDA Version:", torch.version.cuda)
        print("PyTorch cuDNN Version:", torch.backends.cudnn.version())
        print("PyTorch CUDA Device:", torch.cuda.get_device_name(0))
    
    # Print PyTorch CUDA paths
    print("PyTorch CUDA Home:", torch.utils.cpp_extension.CUDA_HOME)
except Exception as e:
    print("PyTorch Initialization Error:", e)

