"""
工具
"""

import shutil
import sys
import time
import uuid
from collections import OrderedDict

import requests

from .constants import *
from .utils import Logger

logger = Logger.get_logger()


class LimitedSizeDict(OrderedDict):
    """
    带有限制大小的字典
    """
    def __init__(self, max_size):
        self._max_size = max_size
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        elif len(self) >= self._max_size:
            oldest_key = next(iter(self))
            del self[oldest_key]
        super().__setitem__(key, value)


def restart() -> None:
    """
    MRB2重启
    Returns:
        None
    """
    # 获取当前解释器路径
    p = sys.executable
    try:
        # 启动新程序(解释器路径, 当前程序)
        os.execl(p, p, *sys.argv)
    except OSError:
        # 关闭当前程序
        sys.exit()


def download_file_to_cache(url: str, headers=None, file_name: str = "",
                           download_path: str = None, stream=False, fake_headers: bool = True) -> str | None:
    """
    下载文件到缓存
    Args:
        url: 下载的url
        headers: 下载请求的请求头
        file_name: 文件名
        download_path: 下载路径
        stream: 是否使用流式传输
        fake_headers: 是否使用自动生成的假请求头
    Returns:
        文件路径
    """
    if headers is None:
        headers = {}

    if fake_headers:
        headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42")
        headers["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.8,da;q=0.7,ko;q=0.6"
        headers["Accept-Encoding"] = "gzip, deflate, br"
        headers["Accept"] = ("text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,"
                             "application/signed-exchange;v=b3;q=0.7")
        headers["Connection"] = "keep-alive"
        headers["Upgrade-Insecure-Requests"] = "1"
        headers["Cache-Control"] = "max-age=0"
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = "none"
        headers["Sec-Fetch-User"] = "?1"
        headers["Sec-Ch-Ua"] = "\"Chromium\";v=\"113\", \"Not-A.Brand\";v=\"24\", \"Microsoft Edge\";v=\"113\""
        headers["Sec-Ch-Ua-Mobile"] = "?0"
        headers["Sec-Ch-Ua-Platform"] = "\"Windows\""
        headers["Host"] = url.split("/")[2]

    # 路径拼接
    if file_name == "":
        file_name = uuid.uuid4().hex + ".cache"

    if download_path is None:
        file_path = os.path.join(CACHE_PATH, file_name)
    else:
        file_path = os.path.join(download_path, file_name)

    # 路径不存在特判
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    try:
        # 下载
        if stream:
            with open(file_path, "wb") as f, requests.get(url, stream=True, headers=headers) as res:
                for chunk in res.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        break
                    f.write(chunk)
        else:
            # 不使用流式传输
            res = requests.get(url, headers=headers)

            with open(file_path, "wb") as f:
                f.write(res.content)
    except requests.exceptions.RequestException as e:
        logger.warning(f"下载文件失败: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return None

    return file_path


# 删除缓存文件
def clean_cache() -> None:
    """
    清理缓存
    Returns:
        None
    """
    if os.path.exists(CACHE_PATH):
        try:
            shutil.rmtree(CACHE_PATH, ignore_errors=True)
        except Exception as e:
            logger.warning("删除缓存时报错，报错信息: %s" % repr(e))


# 函数缓存
def function_cache(max_size: int, expiration_time: int = -1):
    """
    函数缓存
    Args:
        max_size: 最大大小
        expiration_time: 过期时间
    Returns:
        None
    """
    cache = LimitedSizeDict(max_size)

    def cache_decorator(func):
        """
        缓存装饰器
        Args:
            @param func:
        Returns:
            None
        """
        def wrapper(*args, **kwargs):
            key = str(func.__name__) + str(args) + str(kwargs)
            if key in cache and (expiration_time == -1 or time.time() - cache[key][1] < expiration_time):
                return cache[key][0]
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        def clear_cache():
            """清理缓存"""
            cache.clear()

        def get_cache():
            """获取缓存"""
            return dict(cache)

        def original_func(*args, **kwargs):
            """调用原函数"""
            return func(*args, **kwargs)

        wrapper.clear_cache = clear_cache
        wrapper.get_cache = get_cache
        wrapper.original_func = original_func
        return wrapper

    return cache_decorator
