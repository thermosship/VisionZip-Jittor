# AutoDL环境使用说明

## 每次有卡开机后

```bash
source /root/autodl-tmp/VisionZip-Jittor/environment/activate_jittor.sh
```

脚本会激活：

```text
/root/autodl-tmp/envs/visionzip-jittor
```

并进入项目目录。

## 首次保存环境证据

```bash
bash environment/collect_env.sh
```

生成内容位于`environment/generated/`。这些原始文件默认不提交；README中只整理必要版本和结论。

## 当前已验证环境

- Ubuntu 22.04.1 LTS
- Python 3.10.8
- Jittor 1.3.11.0
- CUDA Toolkit 11.8.89
- RTX 4090 24GB，CUDA架构`sm_89`
- Jittor官方CUDA测试通过
