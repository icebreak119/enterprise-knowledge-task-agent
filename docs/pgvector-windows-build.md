# 在 Windows 上为 PostgreSQL 编译安装 pgvector

记录本机（PostgreSQL 17.2 + VS 2022 Community）编译安装 pgvector 的完整过程与踩过的坑。
当 Docker 不可用时，用这份文档在本机还原向量检索能力。

**当前结果**：pgvector **0.7.4**，`vector.dll` 已装入 `C:\Program Files\PostgreSQL\17\lib`。

---

## 1. 准备

| 依赖 | 说明 |
|---|---|
| PostgreSQL 17 | 官方安装包自带 `include\server` 与 `lib\postgres.lib`，扩展编译必需 |
| Visual Studio 2022 | 需要 MSVC 的 `cl.exe` / `nmake.exe` / `link.exe` |
| Windows SDK | 提供 `corecrt.h` 等头文件，需手动加入 `INCLUDE` |

```bash
git clone --depth 1 --branch v0.7.4 https://github.com/pgvector/pgvector
```

## 2. 编译

### 2.1 复制工具链到无空格目录

`nmake` 展开 `CC` 宏时会吃掉引号，`C:\Program Files\...` 会被空格截断。
把 MSVC 工具链复制到无空格路径绕开：

```powershell
$src = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
Copy-Item "$src\*" "C:\Users\31368\Desktop\rag\vctools" -Recurse -Force
```

### 2.2 设置编译环境

不要用 `vcvars64.bat`（会踩到下面的坑 1），直接手工设置：

```powershell
$env:INCLUDE = "<msvc>\include;<sdk>\Include\<ver>\ucrt;<sdk>\Include\<ver>\um;<sdk>\Include\<ver>\shared" -join ';'
$env:LIB     = "<msvc>\lib\x64;<sdk>\Lib\<ver>\ucrt\x64;<sdk>\Lib\<ver>\um\x64;C:\Program Files\PostgreSQL\17\lib" -join ';'
$env:PGROOT  = "C:\Program Files\PostgreSQL\17"
```

### 2.3 补丁：补齐 PG17 未导出的符号

PostgreSQL 17 的 Windows 构建**没有导出** `common/shortest_dec.h` 中声明的
`float_to_shortest_decimal_bufn` / `float_to_shortest_decimal_buf`，链接时必报：

```
halfvec.obj : error LNK2019: 无法解析的外部符号 float_to_shortest_decimal_bufn
```

在 `src/` 下新增 `wincompat.c` 提供最短往返精度实现，并把它加进 `Makefile.win` 的 `OBJS`：

```c
static int shortest_decimal_buf(float f, char *result)
{
    char tmp[64]; int n = 0;
    for (int prec = 1; prec <= 9; prec++) {
        n = sprintf(tmp, "%.*g", prec, (double) f);
        if ((float) strtod(tmp, NULL) == f) break;
    }
    memcpy(result, tmp, n + 1);
    return n;
}
int float_to_shortest_decimal_bufn(float f, char *result) { return shortest_decimal_buf(f, result); }
int float_to_shortest_decimal_buf(float f, char *result)  { return shortest_decimal_buf(f, result); }
```

> 0.8.6 同样依赖这两个符号，降级不能解决问题，必须打补丁。

### 2.4 编译

```powershell
Start-Process -FilePath "C:\...\vctools\nmake.exe" `
  -ArgumentList "/f Makefile.win CC=C:\...\vctools\cl.exe" `
  -WorkingDirectory "<源码目录>" -NoNewWindow -Wait -PassThru `
  -RedirectStandardOutput out.log -RedirectStandardError err.log
```

产物：`vector.dll`。

## 3. 安装（需要管理员权限）

```powershell
Start-Process robocopy.exe -Verb RunAs -Wait -PassThru -ArgumentList `
  '"<源码目录>" "C:\Program Files\PostgreSQL\17\lib" vector.dll'
Start-Process robocopy.exe -Verb RunAs -Wait -PassThru -ArgumentList `
  '"<源码目录>" "C:\Program Files\PostgreSQL\17\share\extension" vector.control'
Start-Process robocopy.exe -Verb RunAs -Wait -PassThru -ArgumentList `
  '"<源码目录>\sql" "C:\Program Files\PostgreSQL\17\share\extension" vector--0.7.4.sql'
```

> `robocopy` 退出码 **1 表示复制成功**（不是错误）。
> SQL 脚本在 `sql/` 子目录，别漏掉，否则报 `没有安装脚本`。

## 4. 验证

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname = 'vector';   -- 0.7.4
SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector;                 -- 向量算子可用
```

## 5. 踩坑清单

| 坑 | 现象 | 解法 |
|---|---|---|
| 代理变量大小写冲突 | `Enter-VsDevShell` 与 `Start-Process` 抛"已添加项。字典中的关键字: http_proxy" | 启动前 `Remove-Item Env:\http_proxy` 等小写变量 |
| nmake 吃掉引号 | `'C:\Program' 不是内部或外部命令` | 工具链复制到无空格目录 |
| 缺 SDK 头文件 | `fatal error C1083: 无法打开包括文件: corecrt.h` | `INCLUDE` 手工加上 Windows Kits 路径 |
| 未导出符号 | `LNK2019 float_to_shortest_decimal_bufn` | 加 `wincompat.c` 补丁 |
| 写 Program Files 被拒 | `Permission denied` | `robocopy` + `-Verb RunAs` 提权 |

## 6. 环境限制备忘

- 本环境 `cmd.exe`、`wsl.exe`、`sc.exe`、COM 对象、`[Diagnostics.Process]::Start` 均被安全策略拦截
- PowerShell 工具不回显 stdout，需把输出写入文件再读取
