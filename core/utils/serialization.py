import os
import pickle
from core.tensor import Tensor, zeros, randn
from core.nn.tensor_nn import Module


# def state_dict(self, prefix=''):
#     """
#     返回模块的状态字典，包含所有参数和缓冲区
#
#     Args:
#         prefix (str): 参数名前缀
#
#     Returns:
#         dict: 状态字典
#     """
#     state_dict = {}
#
#     # 收集当前模块的参数和缓冲区
#     for name, value in self.__dict__.items():
#         if isinstance(value, Tensor):
#             # 参数或缓冲区
#             full_name = prefix + name if prefix else name
#             state_dict[full_name] = value.data.copy()
#         elif isinstance(value, Module):
#             # 子模块
#             sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
#             state_dict.update(value.state_dict(sub_prefix))
#         elif isinstance(value, list):
#             # 模块列表（如Sequential中的layers）
#             for i, item in enumerate(value):
#                 if isinstance(item, Module):
#                     sub_prefix = f"{prefix}{name}.{i}." if prefix else f"{name}.{i}."
#                     state_dict.update(item.state_dict(sub_prefix))
#
#     return state_dict
#
#
# def load_state_dict(self, state_dict, strict=True):
#     """
#     从状态字典加载参数
#
#     Args:
#         state_dict (dict): 状态字典
#         strict (bool): 是否严格匹配参数名
#
#     Returns:
#         tuple: (missing_keys, unexpected_keys)
#     """
#     missing_keys = []
#     unexpected_keys = []
#
#     # 获取当前模型的状态字典
#     current_state_dict = self.state_dict()
#
#     # 检查缺失的键
#     for key in current_state_dict.keys():
#         if key not in state_dict:
#             missing_keys.append(key)
#
#     # 检查多余的键
#     for key in state_dict.keys():
#         if key not in current_state_dict:
#             unexpected_keys.append(key)
#
#     # 在严格模式下，如果有缺失或多余的键，抛出异常
#     if strict and (missing_keys or unexpected_keys):
#         error_msg = []
#         if missing_keys:
#             error_msg.append(f"Missing keys: {missing_keys}")
#         if unexpected_keys:
#             error_msg.append(f"Unexpected keys: {unexpected_keys}")
#         raise RuntimeError("Error(s) in loading state_dict:\n" + "\n".join(error_msg))
#
#     # 加载参数
#     self._load_from_state_dict(state_dict, prefix='')
#
#     return missing_keys, unexpected_keys
#
#
# def _load_from_state_dict(self, state_dict, prefix=''):
#     """
#     递归加载状态字典中的参数
#
#     Args:
#         state_dict (dict): 状态字典
#         prefix (str): 当前前缀
#     """
#     # 加载当前模块的参数
#     for name, value in self.__dict__.items():
#         if isinstance(value, Tensor):
#             full_name = prefix + name if prefix else name
#             if full_name in state_dict:
#                 # 检查形状是否匹配
#                 if value.shape != state_dict[full_name].shape:
#                     raise RuntimeError(
#                         f"Size mismatch for {full_name}: "
#                         f"copying a param with shape {state_dict[full_name].shape} "
#                         f"from checkpoint, the shape in current model is {value.shape}"
#                     )
#                 # 加载数据
#                 value.data = state_dict[full_name].copy()
#
#         elif isinstance(value, Module):
#             # 递归加载子模块
#             sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
#             value._load_from_state_dict(state_dict, sub_prefix)
#
#         elif isinstance(value, list):
#             # 处理模块列表
#             for i, item in enumerate(value):
#                 if isinstance(item, Module):
#                     sub_prefix = f"{prefix}{name}.{i}." if prefix else f"{name}.{i}."
#                     item._load_from_state_dict(state_dict, sub_prefix)
#
#
def save(self, filepath):
    """
    保存模型到文件

    Args:
        filepath (str): 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # 保存状态字典
    with open(filepath, 'wb') as f:
        pickle.dump(self.state_dict(), f)

    print(f"模型已保存到: {filepath}")


def load(self, filepath, strict=True):
    """
    从文件加载模型

    Args:
        filepath (str): 模型文件路径
        strict (bool): 是否严格匹配参数名

    Returns:
        tuple: (missing_keys, unexpected_keys)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")

    # 加载状态字典
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)

    # 加载到模型
    missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)

    print(f"模型已从 {filepath} 加载")
    if missing_keys:
        print(f"缺失的键: {missing_keys}")
    if unexpected_keys:
        print(f"多余的键: {unexpected_keys}")

    return missing_keys, unexpected_keys


def named_parameters(self, prefix=''):
    """
    返回模块的所有参数及其名称

    Args:
        prefix (str): 参数名前缀

    Yields:
        tuple: (name, parameter)
    """
    for name, value in self.__dict__.items():
        if isinstance(value, Tensor) and value.requires_grad:
            full_name = prefix + name if prefix else name
            yield full_name, value
        elif isinstance(value, Module):
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from value.named_parameters(sub_prefix)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, Module):
                    sub_prefix = f"{prefix}{name}.{i}." if prefix else f"{name}.{i}."
                    yield from item.named_parameters(sub_prefix)


def load(self, filepath, strict=True):
    with open(filepath, 'rb') as f:
        return pickle.load(f)  # 仅仅是标准 pickle

# def load(self, filepath, strict=True):
#     """
#     从文件加载模型
#
#     Args:
#         filepath (str): 模型文件路径
#         strict (bool): 是否严格匹配参数名
#
#     Returns:
#         tuple: (missing_keys, unexpected_keys)
#     """
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"模型文件不存在: {filepath}")
#
#     # 加载状态字典
#     with open(filepath, 'rb') as f:
#         state_dict = pickle.load(f)
#
#     # 加载到模型
#     missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
#
#     print(f"模型已从 {filepath} 加载")
#     if missing_keys:
#         print(f"缺失的键: {missing_keys}")
#     if unexpected_keys:
#         print(f"多余的键: {unexpected_keys}")
#
#     return missing_keys, unexpected_keys
