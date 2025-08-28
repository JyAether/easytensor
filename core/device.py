import numpy as np
from typing import Union, Optional


class Device:
    """设备类 - 管理CPU和GPU设备"""

    def __init__(self, device_type: str, index: Optional[int] = None):
        """
        Args:
            device_type: 设备类型 ('cpu', 'cuda')
            index: 设备索引（对于cuda设备）
        """
        device_type = device_type.lower()
        if device_type not in ['cpu', 'cuda']:
            raise ValueError(f"不支持的设备类型: {device_type}")

        self.type = device_type
        self.index = index if index is not None else 0

        # 验证CUDA设备
        if self.type == 'cuda' and not cuda.is_available():
            raise RuntimeError("CUDA不可用，但请求了CUDA设备")

        if self.type == 'cuda' and self.index >= cuda.device_count():
            raise RuntimeError(f"CUDA设备索引 {self.index} 超出范围，可用设备数: {cuda.device_count()}")

    def __str__(self):
        if self.type == 'cuda' and self.index is not None:
            return f"cuda:{self.index}"
        return self.type

    def __repr__(self):
        return f"Device('{self.__str__()}')"

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        elif isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self):
        return hash((self.type, self.index))


class CudaManager:
    """CUDA管理器 - 检测和管理CUDA设备"""

    def __init__(self):
        self._available = None
        self._device_count = None
        self._current_device = 0
        self._check_cuda()

    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import cupy as cp
            # 尝试创建一个简单的数组来测试CUDA
            test_array = cp.array([1, 2, 3])
            del test_array
            self._available = True
            self._device_count = cp.cuda.runtime.getDeviceCount()
        except (ImportError, Exception):
            # 如果cupy不可用或出现其他错误，尝试其他方法
            try:
                import pycuda.driver as cuda_driver
                cuda_driver.init()
                self._available = True
                self._device_count = cuda_driver.Device.count()
            except (ImportError, Exception):
                # 最后尝试通过nvidia-ml-py检查
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    self._device_count = pynvml.nvmlDeviceGetCount()
                    self._available = True
                except (ImportError, Exception):
                    self._available = False
                    self._device_count = 0

    def is_available(self) -> bool:
        """检查CUDA是否可用"""
        return self._available

    def device_count(self) -> int:
        """返回可用的CUDA设备数量"""
        return self._device_count if self._available else 0

    def current_device(self) -> int:
        """返回当前设备索引"""
        return self._current_device

    def set_device(self, device: Union[int, str, Device]):
        """设置当前CUDA设备"""
        if not self._available:
            raise RuntimeError("CUDA不可用")

        if isinstance(device, Device):
            device_index = device.index
        elif isinstance(device, str):
            if device.startswith('cuda:'):
                device_index = int(device.split(':')[1])
            else:
                raise ValueError(f"无效的设备字符串: {device}")
        elif isinstance(device, int):
            device_index = device
        else:
            raise ValueError(f"不支持的设备类型: {type(device)}")

        if device_index >= self._device_count:
            raise ValueError(f"设备索引 {device_index} 超出范围，可用设备数: {self._device_count}")

        self._current_device = device_index

        # 如果有cupy，设置当前设备
        try:
            import cupy as cp
            cp.cuda.Device(device_index).use()
        except ImportError:
            pass

    def get_device_properties(self, device: Union[int, str, Device] = None):
        """获取设备属性"""
        if not self._available:
            raise RuntimeError("CUDA不可用")

        if device is None:
            device_index = self._current_device
        elif isinstance(device, int):
            device_index = device
        elif isinstance(device, str):
            device_index = int(device.split(':')[1]) if device.startswith('cuda:') else 0
        elif isinstance(device, Device):
            device_index = device.index
        else:
            raise ValueError(f"不支持的设备类型: {type(device)}")

        try:
            import cupy as cp
            with cp.cuda.Device(device_index):
                props = cp.cuda.runtime.getDeviceProperties(device_index)
                return {
                    'name': props['name'].decode('utf-8'),
                    'total_memory': props['totalGlobalMem'],
                    'multiprocessor_count': props['multiProcessorCount'],
                    'max_threads_per_block': props['maxThreadsPerBlock'],
                    'max_block_dimensions': props['maxThreadsDim'],
                    'max_grid_dimensions': props['maxGridSize'],
                    'warp_size': props['warpSize'],
                    'compute_capability': (props['major'], props['minor'])
                }
        except ImportError:
            return {'name': f'CUDA Device {device_index}', 'available': True}

    def memory_info(self, device: Union[int, str, Device] = None):
        """获取内存信息"""
        if not self._available:
            raise RuntimeError("CUDA不可用")

        if device is None:
            device_index = self._current_device
        elif isinstance(device, int):
            device_index = device
        elif isinstance(device, str):
            device_index = int(device.split(':')[1]) if device.startswith('cuda:') else 0
        elif isinstance(device, Device):
            device_index = device.index
        else:
            raise ValueError(f"不支持的设备类型: {type(device)}")

        try:
            import cupy as cp
            with cp.cuda.Device(device_index):
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                return {
                    'free': free_mem,
                    'total': total_mem,
                    'used': total_mem - free_mem
                }
        except ImportError:
            return {'free': 0, 'total': 0, 'used': 0}


# 全局CUDA管理器实例
cuda = CudaManager()


def device(device_str: Union[str, Device] = None) -> Device:
    """
    创建设备对象

    Args:
        device_str: 设备字符串 ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc.)
                   如果为None，自动选择最佳设备

    Returns:
        Device: 设备对象

    Examples:
        >>> dev = device('cuda:0')
        >>> dev = device('cpu')
        >>> dev = device()  # 自动选择
    """
    if device_str is None:
        # 自动选择最佳设备
        if cuda.is_available():
            return Device('cuda', 0)
        else:
            return Device('cpu')

    if isinstance(device_str, Device):
        return device_str

    if isinstance(device_str, str):
        device_str = device_str.lower().strip()

        if device_str == 'cpu':
            return Device('cpu')
        elif device_str == 'cuda':
            if not cuda.is_available():
                raise RuntimeError("CUDA不可用")
            return Device('cuda', 0)
        elif device_str.startswith('cuda:'):
            if not cuda.is_available():
                raise RuntimeError("CUDA不可用")
            try:
                index = int(device_str.split(':')[1])
                return Device('cuda', index)
            except (ValueError, IndexError):
                raise ValueError(f"无效的CUDA设备字符串: {device_str}")
        else:
            raise ValueError(f"不支持的设备字符串: {device_str}")

    raise ValueError(f"不支持的设备类型: {type(device_str)}")


def get_default_device() -> Device:
    """获取默认设备（类似PyTorch的自动选择逻辑）"""
    if cuda.is_available():
        return device('cuda:0')
    else:
        return device('cpu')


def set_default_device(dev: Union[str, Device]):
    """设置默认设备"""
    global _default_device
    _default_device = device(dev)
    if dev.type == 'cuda':
        cuda.set_device(dev)


# 全局默认设备
_default_device = get_default_device()


class DeviceContext:
    """设备上下文管理器"""

    def __init__(self, dev: Union[str, Device]):
        self.device = device(dev)
        self.previous_device = None

    def __enter__(self):
        self.previous_device = cuda.current_device() if cuda.is_available() else None
        if self.device.type == 'cuda':
            cuda.set_device(self.device)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_device is not None and cuda.is_available():
            cuda.set_device(self.previous_device)


# ==================== 辅助函数 ====================

def is_available() -> bool:
    """检查CUDA是否可用（兼容PyTorch接口）"""
    return cuda.is_available()


def device_count() -> int:
    """返回可用设备数量"""
    return cuda.device_count()


def current_device() -> int:
    """返回当前设备索引"""
    return cuda.current_device()


def get_device_properties(device_id: Union[int, str, Device] = None):
    """获取设备属性"""
    return cuda.get_device_properties(device_id)


def memory_info(device_id: Union[int, str, Device] = None):
    """获取内存信息"""
    return cuda.memory_info(device_id)


def synchronize():
    """同步所有CUDA操作"""
    if cuda.is_available():
        try:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        except ImportError:
            pass


# ==================== 示例使用 ====================

if __name__ == "__main__":
    print("=== 设备管理系统测试 ===")

    # 检查CUDA可用性
    print(f"CUDA可用: {cuda.is_available()}")
    print(f"CUDA设备数量: {cuda.device_count()}")

    # 创建设备对象
    cpu_device = device('cpu')
    print(f"CPU设备: {cpu_device}")

    if cuda.is_available():
        cuda_device = device('cuda:0')
        print(f"CUDA设备: {cuda_device}")

        # 获取设备属性
        try:
            props = get_device_properties(0)
            print(f"设备属性: {props}")
        except Exception as e:
            print(f"获取设备属性失败: {e}")

        # 获取内存信息
        try:
            mem_info = memory_info(0)
            print(f"内存信息: {mem_info}")
        except Exception as e:
            print(f"获取内存信息失败: {e}")

    # 自动选择设备
    auto_device = device()
    print(f"自动选择的设备: {auto_device}")

    # 设备比较
    print(f"设备相等性测试: {cpu_device == 'cpu'}")

    # 使用设备上下文
    print("\n=== 设备上下文测试 ===")
    with DeviceContext('cpu') as dev:
        print(f"在上下文中使用设备: {dev}")

    print("设备管理系统测试完成！")


# ==================== 与Tensor集成的扩展 ====================

def tensor_to_device_method(self, device_obj):
    """
    为Tensor类添加的to方法扩展
    这个方法应该被添加到您的Tensor类中
    """
    if isinstance(device_obj, str):
        device_obj = device(device_obj)
    elif not isinstance(device_obj, Device):
        raise ValueError(f"不支持的设备类型: {type(device_obj)}")

    if device_obj.type == 'cpu':
        # 转换到CPU
        if hasattr(self, '_cupy_array'):
            # 如果当前数据在GPU上，需要转换到CPU
            import cupy as cp
            self.data = cp.asnumpy(self._cupy_array)
            delattr(self, '_cupy_array')
        self.device = str(device_obj)

    elif device_obj.type == 'cuda':
        # 转换到GPU
        try:
            import cupy as cp
            if not hasattr(self, '_cupy_array'):
                # 如果当前数据在CPU上，转换到GPU
                with cp.cuda.Device(device_obj.index):
                    self._cupy_array = cp.asarray(self.data)
            else:
                # 如果已经在GPU上，但可能在不同设备上
                current_device = self._cupy_array.device.id
                if current_device != device_obj.index:
                    with cp.cuda.Device(device_obj.index):
                        self._cupy_array = cp.asarray(self._cupy_array)

            self.device = str(device_obj)

        except ImportError:
            raise RuntimeError("需要安装CuPy来支持CUDA操作: pip install cupy")

    return self


# 使用示例（集成到您的代码中）
"""
# 在您的主代码中，可以这样使用：

from core.tensor import Tensor
import device_management as dm

# 自动选择最佳设备
device = dm.device('cuda' if dm.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建张量并移动到设备
x = Tensor([1, 2, 3, 4, 5])
x = x.to(device)

# 创建模型并移动到设备
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)
model = model.to(device)

# 在训练循环中
for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # 前向传播
    output = model(batch_x)
    loss = criterion(output, batch_y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""