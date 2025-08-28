import os
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any


class ModelIO:
    """模型输入输出管理器 - 提供保存和加载模型的功能"""

    @staticmethod
    def save(state_dict: Dict[str, np.ndarray], filepath: Union[str, Path]):
        """
        保存模型状态字典到文件

        Args:
            state_dict: 模型状态字典
            filepath: 保存路径

        Examples:
            >>> save(model.state_dict(), 'model.pth')
            >>> save(model.state_dict(), Path('models') / 'model.pth')
        """
        filepath = Path(filepath)

        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 根据文件扩展名选择保存格式
        if filepath.suffix.lower() in ['.pth', '.pt', '.pkl', '.pickle']:
            # 使用pickle格式（类似PyTorch的.pth）
            ModelIO._save_pickle(state_dict, filepath)
        elif filepath.suffix.lower() == '.npz':
            # 使用numpy的npz格式
            ModelIO._save_npz(state_dict, filepath)
        elif filepath.suffix.lower() == '.json':
            # 使用JSON格式（仅支持小模型）
            ModelIO._save_json(state_dict, filepath)
        else:
            # 默认使用pickle格式
            ModelIO._save_pickle(state_dict, filepath)

        print(f"模型已保存到: {filepath}")

    @staticmethod
    def load(filepath: Union[str, Path], map_location: str = None) -> Dict[str, np.ndarray]:
        """
        从文件加载模型状态字典

        Args:
            filepath: 模型文件路径
            map_location: 设备映射（如'cpu', 'cuda'等）

        Returns:
            Dict[str, np.ndarray]: 状态字典

        Examples:
            >>> state_dict = load('model.pth')
            >>> state_dict = load('model.pth', map_location='cpu')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        # 根据文件扩展名选择加载格式
        if filepath.suffix.lower() in ['.pth', '.pt', '.pkl', '.pickle']:
            state_dict = ModelIO._load_pickle(filepath)
        elif filepath.suffix.lower() == '.npz':
            state_dict = ModelIO._load_npz(filepath)
        elif filepath.suffix.lower() == '.json':
            state_dict = ModelIO._load_json(filepath)
        else:
            # 尝试pickle格式
            try:
                state_dict = ModelIO._load_pickle(filepath)
            except:
                # 如果失败，尝试npz格式
                state_dict = ModelIO._load_npz(filepath)

        # 处理设备映射
        if map_location is not None:
            # 这里可以添加设备转换逻辑
            pass

        print(f"模型已从 {filepath} 加载")
        return state_dict

    @staticmethod
    def _save_pickle(state_dict: Dict[str, np.ndarray], filepath: Path):
        """使用pickle格式保存"""
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_pickle(filepath: Path) -> Dict[str, np.ndarray]:
        """使用pickle格式加载"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _save_npz(state_dict: Dict[str, np.ndarray], filepath: Path):
        """使用numpy npz格式保存"""
        np.savez_compressed(filepath, **state_dict)

    @staticmethod
    def _load_npz(filepath: Path) -> Dict[str, np.ndarray]:
        """使用numpy npz格式加载"""
        data = np.load(filepath)
        return {key: data[key] for key in data.keys()}

    @staticmethod
    def _save_json(state_dict: Dict[str, np.ndarray], filepath: Path):
        """使用JSON格式保存（仅适用于小模型）"""
        json_data = {}
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                json_data[key] = {
                    'data': value.tolist(),
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            else:
                json_data[key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_json(filepath: Path) -> Dict[str, np.ndarray]:
        """使用JSON格式加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        state_dict = {}
        for key, value in json_data.items():
            if isinstance(value, dict) and 'data' in value:
                # 重构numpy数组
                data = np.array(value['data'], dtype=value['dtype'])
                state_dict[key] = data.reshape(value['shape'])
            else:
                state_dict[key] = value

        return state_dict


# ==================== 兼容PyTorch接口的函数 ====================

def save(obj: Union[Dict[str, np.ndarray], Any], filepath: Union[str, Path]):
    """
    保存对象到文件（兼容PyTorch torch.save接口）

    Args:
        obj: 要保存的对象（通常是model.state_dict()）
        filepath: 保存路径

    Examples:
        >>> save(model.state_dict(), 'model.pth')
        >>> save(model.state_dict(), config.MODELS_DIR / 'model.pth')
    """
    if hasattr(obj, 'items') and callable(obj.items):
        # 如果是字典类型（state_dict）
        ModelIO.save(obj, filepath)
    else:
        # 如果是其他对象，直接用pickle保存
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"对象已保存到: {filepath}")


def load(filepath: Union[str, Path], map_location: str = None):
    """
    从文件加载对象（兼容PyTorch torch.load接口）

    Args:
        filepath: 文件路径
        map_location: 设备映射

    Returns:
        加载的对象

    Examples:
        >>> state_dict = load('model.pth')
        >>> model.load_state_dict(load('model.pth'))
        >>> state_dict = load(config.MODELS_DIR / 'model.pth')
    """
    return ModelIO.load(filepath, map_location)


# ==================== 模型检查点功能 ====================

class Checkpoint:
    """模型检查点管理器"""

    def __init__(self, model, optimizer=None, scheduler=None):
        """
        Args:
            model: 模型对象
            optimizer: 优化器对象（可选）
            scheduler: 学习率调度器对象（可选）
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save(self, filepath: Union[str, Path], epoch: int = None, loss: float = None,
             accuracy: float = None, **kwargs):
        """
        保存完整的检查点

        Args:
            filepath: 保存路径
            epoch: 当前轮数
            loss: 当前损失
            accuracy: 当前准确率
            **kwargs: 其他要保存的信息
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
        }

        if self.optimizer is not None:
            # 如果优化器有state_dict方法
            if hasattr(self.optimizer, 'state_dict'):
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            else:
                # 简单保存优化器参数
                checkpoint['optimizer_params'] = {
                    'lr': getattr(self.optimizer, 'lr', None),
                    'momentum': getattr(self.optimizer, 'momentum', None),
                    'weight_decay': getattr(self.optimizer, 'weight_decay', None),
                }

        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 添加额外信息
        checkpoint.update(kwargs)

        save(checkpoint, filepath)

    def load(self, filepath: Union[str, Path], load_optimizer: bool = True,
             load_scheduler: bool = True, map_location: str = None):
        """
        加载检查点

        Args:
            filepath: 文件路径
            load_optimizer: 是否加载优化器状态
            load_scheduler: 是否加载调度器状态
            map_location: 设备映射

        Returns:
            dict: 检查点信息
        """
        checkpoint = load(filepath, map_location)

        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if load_optimizer and self.optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                if hasattr(self.optimizer, 'load_state_dict'):
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            elif 'optimizer_params' in checkpoint:
                # 恢复简单参数
                params = checkpoint['optimizer_params']
                for key, value in params.items():
                    if hasattr(self.optimizer, key) and value is not None:
                        setattr(self.optimizer, key, value)

        # 加载调度器状态
        if load_scheduler and self.scheduler is not None:
            if 'scheduler_state_dict' in checkpoint:
                if hasattr(self.scheduler, 'load_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint


# ==================== 配置管理 ====================

class Config:
    """配置管理器"""

    def __init__(self, **kwargs):
        """
        创建配置对象

        Examples:
            >>> config = Config(MODELS_DIR=Path('models'), BATCH_SIZE=32)
            >>> config.MODELS_DIR = Path('models')
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, filepath: Union[str, Path]):
        """保存配置到文件"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def load(self, filepath: Union[str, Path]):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            setattr(self, key, value)


# ==================== 示例使用 ====================

if __name__ == "__main__":
    # 导入必要的模块（假设已经实现）
    from pathlib import Path

    # 示例1: 创建配置
    print("=== 配置管理示例 ===")
    config = Config(
        MODELS_DIR=Path('models'),
        DATA_DIR=Path('data'),
        BATCH_SIZE=32,
        LEARNING_RATE=0.001,
        EPOCHS=100
    )

    print(f"模型目录: {config.MODELS_DIR}")
    print(f"批次大小: {config.BATCH_SIZE}")

    # 示例2: 模拟模型状态字典
    print("\n=== 模型保存/加载示例 ===")

    # 模拟一个简单的状态字典
    mock_state_dict = {
        'layer1.weight': np.random.randn(10, 5).astype(np.float32),
        'layer1.bias': np.random.randn(10).astype(np.float32),
        'layer2.weight': np.random.randn(1, 10).astype(np.float32),
        'layer2.bias': np.random.randn(1).astype(np.float32),
    }

    # 保存模型
    models_dir = Path('test_models')
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / 'test_model.pth'
    save(mock_state_dict, model_path)

    # 加载模型
    loaded_state_dict = load(model_path)

    print(f"原始状态字典键: {list(mock_state_dict.keys())}")
    print(f"加载状态字典键: {list(loaded_state_dict.keys())}")
    print(f"权重形状匹配: {mock_state_dict['layer1.weight'].shape == loaded_state_dict['layer1.weight'].shape}")

    # 示例3: 不同格式保存
    print("\n=== 不同格式保存示例 ===")

    # NPZ格式
    save(mock_state_dict, models_dir / 'test_model.npz')
    loaded_npz = load(models_dir / 'test_model.npz')
    print(f"NPZ格式加载成功: {len(loaded_npz) == len(mock_state_dict)}")

    # JSON格式（小模型）
    small_state_dict = {
        'small_weight': np.array([[1, 2], [3, 4]], dtype=np.float32),
        'small_bias': np.array([0.1, 0.2], dtype=np.float32)
    }
    save(small_state_dict, models_dir / 'small_model.json')
    loaded_json = load(models_dir / 'small_model.json')
    print(f"JSON格式加载成功: {len(loaded_json) == len(small_state_dict)}")

    # 示例4: 检查点功能
    print("\n=== 检查点功能示例 ===")


    # 模拟模型和优化器
    class MockModel:
        def state_dict(self):
            return mock_state_dict

        def load_state_dict(self, state_dict):
            print(f"加载了 {len(state_dict)} 个参数")


    class MockOptimizer:
        def __init__(self):
            self.lr = 0.001
            self.momentum = 0.9


    mock_model = MockModel()
    mock_optimizer = MockOptimizer()

    # 创建检查点管理器
    checkpoint = Checkpoint(mock_model, mock_optimizer)

    # 保存检查点
    checkpoint.save(
        models_dir / 'checkpoint.pth',
        epoch=50,
        loss=0.1234,
        accuracy=0.95
    )

    # 加载检查点
    loaded_checkpoint = checkpoint.load(models_dir / 'checkpoint.pth')
    print(f"检查点信息: epoch={loaded_checkpoint['epoch']}, loss={loaded_checkpoint['loss']}")

    # 清理测试文件
    import shutil

    if models_dir.exists():
        shutil.rmtree(models_dir)

    print("\n模型保存/加载系统测试完成！")