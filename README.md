# Minecraft Chunk Triager

用机器学习判断 Java 版存档里哪些区块含有玩家建造的内容，用来在版本升级前清理无用的自然区块。目前支持 Java 26.2 的主世界。

## 准备

- Python 3.12 和 [uv](https://docs.astral.sh/uv/)
- [MCA Selector](https://github.com/Querz/mcaselector)：用于绘制标注网页的底图，可选
- 先用游戏的 forceUpgrade 把存档升级到 26.2，然后**使用存档副本**，不要用正在玩的存档

```
git clone https://github.com/Blind-Guess-Senior/MinecraftChunkTriager.git
cd MinecraftChunkTriager
uv sync
```

## 工作目录

一个工作目录对应一套训练设置，可以放在任何位置。目录里的内容：

```
chunktriager.toml     用哪些存档、每个存档的用途、工具路径
labels/               标注，唯一需要备份的数据
cache/                特征、底图、预测，删掉可以重新生成
models/current/       训练出的模型
```

`chunktriager.toml` 示例（相对路径以工作目录为基准）：

```toml
[mcaselector]
command = ["java", "-jar", "D:/MCA Selector/mcaselector.jar"]

[[worlds]]
name = "main"               # 命令里用这个名字指代存档
path = "D:/saves/MyWorld"
role = "train"

[[worlds]]
name = "friend"
path = "D:/saves/FriendWorld"
role = "val"
```

`role` 的取值：

| role | 用途 |
|---|---|
| train | 用来训练，`evaluate` 时做交叉验证 |
| val | 不参与训练，`evaluate` 时检验模型 |
| test | 最终检验，只有 `evaluate --test` 才用 |
| none | 只提取特征和标注，模型不用 |

`init` 生成的模板里还有训练参数、保护半径等可选项，都带注释。

## 命令

所有命令都在仓库目录下运行，用 `-w` 指定工作目录；不写 `-w` 时用当前目录。

| 命令 | 作用 |
|---|---|
| `init [--worlds <存档文件夹>]` | 创建工作目录和配置模板；`--worlds` 会把文件夹里的每个存档都列进去 |
| `worlds` | 列出存档的地形类型、是否已准备、标注数量 |
| `prepare [名称...]` | 提取每个区块的特征，只重新处理有变化的 region |
| `render [名称...]` | 用 MCA Selector 绘制底图 |
| `label <名称>` | 打开某个存档的标注网页 |
| `train` | 用 role 为 train 的存档训练模型 |
| `evaluate [--test]` | 交叉验证，并在 val（和 test）存档上检验 |
| `predict [名称...] [--model <目录>]` | 用模型给区块打分，结果会显示在标注网页上 |

不写名称时，命令处理全部存档。

## 典型流程

```
uv run chunktriager -w D:/triage init --worlds D:/saves
# 编辑 D:/triage/chunktriager.toml，设置 role 和 MCA Selector 路径
uv run chunktriager -w D:/triage prepare
uv run chunktriager -w D:/triage render
uv run chunktriager -w D:/triage label main
uv run chunktriager -w D:/triage train
uv run chunktriager -w D:/triage evaluate
uv run chunktriager -w D:/triage predict
```

## 标注

标注网页的地址是 `http://127.0.0.1:8765/`。它只监听本机，没有登录验证，不要改成对外开放。

- 只需要标"人工"区块，即区块本身含有玩家建造的内容。未标注的区块在训练时按自然处理。
- 拖动涂抹，Shift + 拖动画矩形，Alt + 点击查看区块信息，Ctrl + Z 撤销。
- 超平坦和虚空存档会显示"基底提议"：生成器层以外有内容的区块标成黄色。检查之后，可以一次把它们写成人工标签。
- 运行 `predict` 之后，刷新网页可以看到模型概率和保护范围。
