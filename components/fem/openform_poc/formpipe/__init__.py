# -*- coding: utf-8 -*-
"""formpipe — 声明式冲压成形仿真管线。

四层(照搬 mbs_isaac 的分层动机:错误在最便宜的地方就被拦住):

    ① 声明  spec.py       纯数据类描述:材料/坯料/工具/工艺
    ② 校验  spec.validate() 纯 numpy,毫秒级,不碰求解器
    ③ 构建  openradioss.write_deck()  spec -> 求解器输入
    ④ 求解  openradioss.run/parse     -> 逐单元减薄 + 判据

求解器是可替换后端:任何实现 write_deck/run/parse 三个函数的模块
都能顶替 openradioss(将来的 Newton / warp.fem 后端由此接入),
agent 只跟 StampingSpec 和结果 dict 打交道。
"""
from .spec import (
    MaterialSpec,
    MeshSpec,
    ProcessSpec,
    SpecError,
    StampingSpec,
    ToolSpec,
)

__all__ = [
    "MaterialSpec", "MeshSpec", "ProcessSpec", "SpecError", "StampingSpec", "ToolSpec",
]
