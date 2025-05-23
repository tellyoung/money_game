import pickle
import os


def save_factor_logic(factor_logic, file_path):
    """
    保存因子计算逻辑到指定文件。

    参数:
        factor_logic: 因子计算逻辑对象。
        file_path: 保存文件的路径。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(factor_logic, f)


def load_factor_logic(file_path):
    """
    从指定文件加载因子计算逻辑。

    参数:
        file_path: 加载文件的路径。

    返回:
        加载的因子计算逻辑对象。
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)