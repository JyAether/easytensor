import numpy as np
import gc
import psutil
import os
from typing import Union, Dict, Any, Optional
import weakref
import threading

try:
    import cupy as cp
    import cupyx

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class MemoryPool:
    """内存池管理器"""

    def __init__(self, device='cpu', initial_size=1024 * 1024 * 100):  # 100MB
        self.device = device
        self.initial_size = initial_size
        self.allocated_blocks = {}
        self.free_blocks = {}
        self.total_allocated = 0
        self.peak_allocated = 0
        self.lock = threading.Lock()

        if device == 'cuda' and CUDA_AVAILABLE:
            # 使用CuPy的内存池
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()

    def allocate(self, size, dtype=np.float32):
        """分配内存块"""
        with self.lock:
            byte_size = size * np.dtype(dtype).itemsize

            if self.device == 'cuda' and CUDA_AVAILABLE:
                # GPU内存分配
                ptr = cp.cuda.alloc(byte_size)
                self.allocated_blocks[id(ptr)] = (ptr, byte_size)
                self.total_allocated += byte_size
                self.peak_allocated = max(self.peak_allocated, self.total_allocated)
                return ptr
            else:
                # CPU内存分配
                array = np.empty(size, dtype=dtype)
                self.allocated_blocks[id(array)] = (array, byte_size)
                self.total_allocated += byte_size
                self.peak_allocated = max(self.peak_allocated, self.total_allocated)
                return array

    def deallocate(self, ptr):
        """释放内存块"""
        with self.lock:
            ptr_id = id(ptr)
            if ptr_id in self.allocated_blocks:
                _, byte_size = self.allocated_blocks[ptr_id]
                del self.allocated_blocks[ptr_id]
                self.total_allocated -= byte_size

    def get_memory_info(self):
        """获取内存使用信息"""
        with self.lock:
            info = {
                'device': self.device,
                'total_allocated': self.total_allocated,
                'peak_allocated': self.peak_allocated,
                'num_blocks': len(self.allocated_blocks)
            }

            if self.device == 'cuda' and CUDA_AVAILABLE:
                info.update({
                    'gpu_memory_used': self.mempool.used_bytes(),
                    'gpu_memory_total': self.mempool.total_bytes(),
                    'pinned_memory_used': self.pinned_mempool.used_bytes(),
                    'pinned_memory_total': self.pinned_mempool.total_bytes()
                })

            return info

    def clear_cache(self):
        """清理内存缓存"""
        with self.lock:
            if self.device == 'cuda' and CUDA_AVAILABLE:
                self.mempool.free_all_blocks()
                self.pinned_mempool.free_all_blocks()
            gc.collect()


class DataTypeManager:
    """数据类型管理器"""

    # 支持的数据类型映射
    DTYPE_MAP = {
        'float16': np.float16,
        'float32': np.float32,
        'float64': np.float64,
        'int8': np.int8,
        'int16': np.int16,
        'int32': np.int32,
        'int64': np.int64,
        'uint8': np.uint8,
        'uint16': np.uint16,
        'uint32': np.uint32,
        'uint64': np.uint64,
        'bool': np.bool_,
        'complex64': np.complex64,
        'complex128': np.complex128
    }

    @staticmethod
    def get_dtype(dtype_str: Union[str, np.dtype]):
        """获取numpy数据类型"""
        if isinstance(dtype_str, str):
            if dtype_str in DataTypeManager.DTYPE_MAP:
                return DataTypeManager.DTYPE_MAP[dtype_str]
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype_str

    @staticmethod
    def get_dtype_info(dtype):
        """获取数据类型信息"""
        dtype = DataTypeManager.get_dtype(dtype)
        return {
            'name': dtype.name,
            'size': dtype.itemsize,
            'kind': dtype.kind,
            'is_integer': np.issubdtype(dtype, np.integer),
            'is_float': np.issubdtype(dtype, np.floating),
            'is_complex': np.issubdtype(dtype, np.complexfloating),
            'is_signed': np.issubdtype(dtype, np.signedinteger) if np.issubdtype(dtype, np.integer) else None,
            'min_value': np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else np.finfo(dtype).min,
            'max_value': np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.finfo(dtype).max,
        }

    @staticmethod
    def promote_types(dtype1, dtype2):
        """类型提升"""
        dtype1 = DataTypeManager.get_dtype(dtype1)
        dtype2 = DataTypeManager.get_dtype(dtype2)
        return np.promote_types(dtype1, dtype2)

    @staticmethod
    def can_cast(from_dtype, to_dtype, casting='safe'):
        """检查是否可以类型转换"""
        from_dtype = DataTypeManager.get_dtype(from_dtype)
        to_dtype = DataTypeManager.get_dtype(to_dtype)
        return np.can_cast(from_dtype, to_dtype, casting=casting)


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.memory_history = []

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        memory_info = self.process.memory_info()
        cpu_memory = {
            'rss': memory_info.rss,  # 物理内存
            'vms': memory_info.vms,  # 虚拟内存
            'percent': self.process.memory_percent(),
            'available': psutil.virtual_memory().available,
            'total': psutil.virtual_memory().total
        }

        gpu_memory = {}
        if CUDA_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                gpu_memory = {
                    'used': mempool.used_bytes(),
                    'total': mempool.total_bytes(),
                    'free': cp.cuda.Device().mem_info[0],  # 空闲GPU内存
                    'total_device': cp.cuda.Device().mem_info[1]  # GPU总内存
                }
            except Exception as e:
                gpu_memory = {'error': str(e)}

        return {
            'cpu': cpu_memory,
            'gpu': gpu_memory,
            'timestamp': psutil.time.time()
        }

    def record_memory(self):
        """记录内存使用"""
        usage = self.get_memory_usage()
        self.memory_history.append(usage)

        current_memory = usage['cpu']['rss']
        self.peak_memory = max(self.peak_memory, current_memory)

        return usage

    def get_memory_summary(self):
        """获取内存使用摘要"""
        if not self.memory_history:
            return {}

        cpu_usage = [record['cpu']['rss'] for record in self.memory_history]
        return {
            'peak_memory': self.peak_memory,
            'current_memory': cpu_usage[-1] if cpu_usage else 0,
            'average_memory': sum(cpu_usage) / len(cpu_usage),
            'memory_growth': cpu_usage[-1] - cpu_usage[0] if len(cpu_usage) > 1 else 0,
            'num_records': len(self.memory_history)
        }

    def clear_history(self):
        """清空历史记录"""
        self.memory_history.clear()


class TensorRegistry:
    """张量注册表，用于跟踪和管理张量对象"""

    def __init__(self):
        self._tensors = weakref.WeakSet()
        self._lock = threading.Lock()

    def register(self, tensor):
        """注册张量"""
        with self._lock:
            self._tensors.add(tensor)

    def get_all_tensors(self):
        """获取所有活跃的张量"""
        with self._lock:
            return list(self._tensors)

    def get_memory_usage(self):
        """获取所有张量的内存使用情况"""
        with self._lock:
            total_memory = 0
            tensor_count = 0
            device_usage = {}

            for tensor in self._tensors:
                if hasattr(tensor, 'data') and hasattr(tensor.data, 'nbytes'):
                    memory = tensor.data.nbytes
                    total_memory += memory
                    tensor_count += 1

                    device = getattr(tensor, 'device', 'cpu')
                    if device not in device_usage:
                        device_usage[device] = {'count': 0, 'memory': 0}
                    device_usage[device]['count'] += 1
                    device_usage[device]['memory'] += memory

            return {
                'total_memory': total_memory,
                'tensor_count': tensor_count,
                'device_usage': device_usage
            }

    def cleanup_tensors(self, device=None):
        """清理指定设备的张量（主要用于释放GPU内存）"""
        with self._lock:
            tensors_to_remove = []
            for tensor in self._tensors:
                if device is None or getattr(tensor, 'device', 'cpu') == device:
                    if hasattr(tensor, 'data'):
                        del tensor.data
                    if hasattr(tensor, 'grad') and tensor.grad is not None:
                        del tensor.grad
                    tensors_to_remove.append(tensor)

            for tensor in tensors_to_remove:
                self._tensors.discard(tensor)


# 全局实例
_memory_pools = {}
_tensor_registry = TensorRegistry()
_memory_monitor = MemoryMonitor()


def get_memory_pool(device='cpu'):
    """获取内存池"""
    if device not in _memory_pools:
        _memory_pools[device] = MemoryPool(device)
    return _memory_pools[device]


def get_tensor_registry():
    """获取张量注册表"""
    return _tensor_registry


def get_memory_monitor():
    """获取内存监控器"""
    return _memory_monitor


def memory_summary():
    """打印内存使用摘要"""
    monitor = get_memory_monitor()
    registry = get_tensor_registry()

    # 系统内存
    system_usage = monitor.get_memory_usage()
    memory_summary_data = monitor.get_memory_summary()

    # 张量内存
    tensor_usage = registry.get_memory_usage()

    print("=" * 50)
    print("内存使用摘要")
    print("=" * 50)

    # 系统内存
    print("系统内存:")
    print(f"  当前使用: {system_usage['cpu']['rss'] / 1024 ** 2:.2f} MB")
    print(f"  使用百分比: {system_usage['cpu']['percent']:.2f}%")
    print(f"  可用内存: {system_usage['cpu']['available'] / 1024 ** 2:.2f} MB")

    if memory_summary_data:
        print(f"  峰值使用: {memory_summary_data['peak_memory'] / 1024 ** 2:.2f} MB")
        print(f"  平均使用: {memory_summary_data['average_memory'] / 1024 ** 2:.2f} MB")

    # GPU内存
    if system_usage['gpu'] and 'error' not in system_usage['gpu']:
        print("\nGPU内存:")
        gpu = system_usage['gpu']
        print(f"  已使用: {gpu['used'] / 1024 ** 2:.2f} MB")
        print(f"  总计: {gpu['total'] / 1024 ** 2:.2f} MB")
        if 'free' in gpu:
            print(f"  设备空闲: {gpu['free'] / 1024 ** 2:.2f} MB")
            print(f"  设备总计: {gpu['total_device'] / 1024 ** 2:.2f} MB")

    # 张量内存
    print("\n张量内存:")
    print(f"  张量总数: {tensor_usage['tensor_count']}")
    print(f"  总内存: {tensor_usage['total_memory'] / 1024 ** 2:.2f} MB")

    if tensor_usage['device_usage']:
        print("  按设备分布:")
        for device, usage in tensor_usage['device_usage'].items():
            print(f"    {device}: {usage['count']} 个张量, {usage['memory'] / 1024 ** 2:.2f} MB")

    # 内存池信息
    print("\n内存池:")
    for device, pool in _memory_pools.items():
        info = pool.get_memory_info()
        print(f"  {device}:")
        print(f"    已分配: {info['total_allocated'] / 1024 ** 2:.2f} MB")
        print(f"    峰值: {info['peak_allocated'] / 1024 ** 2:.2f} MB")
        print(f"    块数: {info['num_blocks']}")


def clear_memory_cache(device=None):
    """清理内存缓存"""
    if device is None:
        # 清理所有设备
        for pool in _memory_pools.values():
            pool.clear_cache()
        _tensor_registry.cleanup_tensors()
    else:
        # 清理指定设备
        if device in _memory_pools:
            _memory_pools[device].clear_cache()
        _tensor_registry.cleanup_tensors(device)

    gc.collect()
    print(f"已清理{'所有设备' if device is None else device}的内存缓存")


def set_memory_fraction(fraction, device='cuda'):
    """设置GPU内存使用比例"""
    if device == 'cuda' and CUDA_AVAILABLE:
        # 设置CuPy内存池大小限制
        mempool = cp.get_default_memory_pool()
        total_memory = cp.cuda.Device().mem_info[1]
        limit = int(total_memory * fraction)
        mempool.set_limit(size=limit)
        print(f"GPU内存限制设置为 {limit / 1024 ** 2:.2f} MB ({fraction * 100:.1f}%)")
    else:
        print(f"设备 {device} 不支持内存比例设置")


# ==================== 上下文管理器 ====================

class memory_context:
    """内存管理上下文管理器"""

    def __init__(self, device='cpu', clear_on_exit=True, monitor=True):
        self.device = device
        self.clear_on_exit = clear_on_exit
        self.monitor = monitor
        self.initial_memory = None

    def __enter__(self):
        if self.monitor:
            monitor = get_memory_monitor()
            self.initial_memory = monitor.record_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor:
            monitor = get_memory_monitor()
            final_memory = monitor.record_memory()

            if self.initial_memory:
                memory_diff = final_memory['cpu']['rss'] - self.initial_memory['cpu']['rss']
                print(f"内存变化: {memory_diff / 1024 ** 2:+.2f} MB")

        if self.clear_on_exit:
            clear_memory_cache(self.device)


class dtype_context:
    """数据类型上下文管理器"""

    def __init__(self, default_dtype='float32'):
        self.default_dtype = DataTypeManager.get_dtype(default_dtype)
        self.previous_dtype = None

    def __enter__(self):
        # 这里可以设置全局默认数据类型
        # 在实际实现中，可能需要修改Tensor类来支持全局默认类型
        return self.default_dtype

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复之前的默认数据类型
        pass


# ==================== 工具函数 ====================

def sizeof_fmt(num, suffix='B'):
    """格式化字节大小显示"""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_available_memory(device='cpu'):
    """获取可用内存"""
    if device == 'cpu':
        return psutil.virtual_memory().available
    elif device == 'cuda' and CUDA_AVAILABLE:
        return cp.cuda.Device().mem_info[0]  # 空闲GPU内存
    else:
        return 0


def check_memory_requirements(tensor_shapes, dtype='float32', device='cpu'):
    """检查内存需求"""
    dtype_obj = DataTypeManager.get_dtype(dtype)
    total_elements = sum(np.prod(shape) for shape in tensor_shapes)
    required_memory = total_elements * dtype_obj.itemsize
    available_memory = get_available_memory(device)

    return {
        'required_memory': required_memory,
        'available_memory': available_memory,
        'sufficient': required_memory <= available_memory,
        'memory_ratio': required_memory / available_memory if available_memory > 0 else float('inf')
    }


def optimize_memory_layout(tensor, contiguous=True):
    """优化内存布局"""
    if hasattr(tensor, 'data'):
        xp = tensor._get_array_module() if hasattr(tensor, '_get_array_module') else np

        # 检查是否需要重新排列
        if contiguous and not tensor.data.flags.c_contiguous:
            tensor.data = xp.ascontiguousarray(tensor.data)
            print("已将张量转换为连续内存布局")

        return tensor.data.flags.c_contiguous

    return False


def profile_memory_usage(func):
    """内存使用分析装饰器"""

    def wrapper(*args, **kwargs):
        monitor = get_memory_monitor()

        # 记录开始状态
        start_memory = monitor.record_memory()

        try:
            result = func(*args, **kwargs)
        finally:
            # 记录结束状态
            end_memory = monitor.record_memory()

            memory_diff = end_memory['cpu']['rss'] - start_memory['cpu']['rss']
            print(f"函数 {func.__name__} 内存使用: {sizeof_fmt(memory_diff)}")

            if CUDA_AVAILABLE and 'used' in end_memory['gpu']:
                gpu_diff = end_memory['gpu']['used'] - start_memory['gpu']['used']
                print(f"GPU内存使用: {sizeof_fmt(gpu_diff)}")

        return result

    return wrapper


# ==================== 自动内存管理 ====================

class AutoMemoryManager:
    """自动内存管理器"""

    def __init__(self, max_memory_ratio=0.8, cleanup_threshold=0.9):
        self.max_memory_ratio = max_memory_ratio
        self.cleanup_threshold = cleanup_threshold
        self.enabled = True

    def check_and_cleanup(self, device='cpu'):
        """检查内存使用并在必要时清理"""
        if not self.enabled:
            return

        if device == 'cpu':
            memory = psutil.virtual_memory()
            usage_ratio = (memory.total - memory.available) / memory.total
        elif device == 'cuda' and CUDA_AVAILABLE:
            free_mem, total_mem = cp.cuda.Device().mem_info
            usage_ratio = (total_mem - free_mem) / total_mem
        else:
            return

        if usage_ratio > self.cleanup_threshold:
            print(f"内存使用率 {usage_ratio:.2%} 超过阈值，开始清理...")
            clear_memory_cache(device)

            # 再次检查
            if device == 'cpu':
                memory = psutil.virtual_memory()
                new_usage_ratio = (memory.total - memory.available) / memory.total
            elif device == 'cuda' and CUDA_AVAILABLE:
                free_mem, total_mem = cp.cuda.Device().mem_info
                new_usage_ratio = (total_mem - free_mem) / total_mem

            print(f"清理后内存使用率: {new_usage_ratio:.2%}")

    def enable(self):
        """启用自动内存管理"""
        self.enabled = True

    def disable(self):
        """禁用自动内存管理"""
        self.enabled = False


# 全局自动内存管理器
_auto_memory_manager = AutoMemoryManager()


def get_auto_memory_manager():
    """获取自动内存管理器"""
    return _auto_memory_manager


# ==================== 示例和测试 ====================

def test_memory_management():
    """测试内存管理功能"""
    print("开始内存管理测试...")

    # 测试数据类型管理
    print("\n1. 数据类型测试:")
    dtype_info = DataTypeManager.get_dtype_info('float32')
    print(f"float32 信息: {dtype_info}")

    promoted = DataTypeManager.promote_types('float32', 'int64')
    print(f"float32 + int64 = {promoted}")

    # 测试内存监控
    print("\n2. 内存监控测试:")
    monitor = get_memory_monitor()
    initial_usage = monitor.record_memory()
    print(f"初始内存使用: {sizeof_fmt(initial_usage['cpu']['rss'])}")

    # 创建一些大数组来测试内存使用
    arrays = []
    for i in range(5):
        arr = np.random.randn(1000, 1000).astype(np.float32)
        arrays.append(arr)

    final_usage = monitor.record_memory()
    print(f"创建数组后内存使用: {sizeof_fmt(final_usage['cpu']['rss'])}")
    print(f"内存增长: {sizeof_fmt(final_usage['cpu']['rss'] - initial_usage['cpu']['rss'])}")

    # 测试内存池
    print("\n3. 内存池测试:")
    pool = get_memory_pool('cpu')
    allocated_array = pool.allocate(1000000, np.float32)
    pool_info = pool.get_memory_info()
    print(f"内存池信息: {pool_info}")

    # 清理
    del arrays, allocated_array
    pool.clear_cache()

    print("\n4. 内存摘要:")
    memory_summary()

    print("\n内存管理测试完成!")


def benchmark_dtype_performance():
    """基准测试不同数据类型的性能"""
    print("数据类型性能基准测试...")

    size = (1000, 1000)
    dtypes = ['float16', 'float32', 'float64']

    for dtype_str in dtypes:
        dtype = DataTypeManager.get_dtype(dtype_str)

        import time
        start_time = time.time()

        # 创建数组
        a = np.random.randn(*size).astype(dtype)
        b = np.random.randn(*size).astype(dtype)

        # 执行运算
        c = a @ b
        result = c.sum()

        end_time = time.time()

        memory_usage = a.nbytes + b.nbytes + c.nbytes

        print(f"{dtype_str}:")
        print(f"  时间: {end_time - start_time:.4f}s")
        print(f"  内存: {sizeof_fmt(memory_usage)}")
        print(f"  结果: {result:.6f}")


if __name__ == "__main__":
    # 运行测试
    test_memory_management()
    print("\n" + "=" * 50)
    benchmark_dtype_performance()

    # 演示上下文管理器
    print("\n" + "=" * 50)
    print("上下文管理器演示:")

    with memory_context(monitor=True, clear_on_exit=True):
        print("在内存上下文中创建大数组...")
        big_array = np.random.randn(2000, 2000)
        print(f"数组大小: {sizeof_fmt(big_array.nbytes)}")

    print("上下文结束，内存已清理")

    # 演示自动内存管理
    print("\n自动内存管理演示:")
    auto_manager = get_auto_memory_manager()
    auto_manager.check_and_cleanup('cpu')