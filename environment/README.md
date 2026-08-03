# AutoDL 环境使用说明

## 每次实例开机后

```bash
source /root/autodl-tmp/VisionZip-Jittor/environment/activate_jittor.sh
```

脚本会激活：

```text
/root/autodl-tmp/envs/visionzip-jittor
```

并进入项目目录。

## 保存环境证据

```bash
bash environment/collect_env.sh
```

生成内容位于 `environment/generated/`。这些原始文件默认不提交；README 和阶段证据包记录必要版本与结论。

## 当前已验证环境

- Ubuntu 22.04.1 LTS；
- Jittor 环境 Python 3.10.20；
- Jittor 1.3.11.0；
- PyTorch reference 2.1.2+cu118；
- Transformers 4.31.0；
- Jittor 实际使用 CUDA Toolkit 11.8.89；
- RTX 4090 24GB，CUDA 架构 `sm_89`；
- Jittor 官方 CUDA 测试通过。

`nvidia-smi` 中的 CUDA Version 表示驱动兼容上限，不等于 Jittor 编译所用 Toolkit；应以 `nvcc --version` 和 Jittor CUDA cache key 为准。
