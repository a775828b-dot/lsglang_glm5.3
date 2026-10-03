"""Collect GLM-5.3 outputs (reasoning + answer) for the draft hot-token map.
Usage: gen_corpus.py PORT OUT.txt  — diverse agent/coding/Chinese/English/tool prompts, half greedy."""
import json, sys, urllib.request

port, out = int(sys.argv[1]), sys.argv[2]
P = [
 '用 Python 写一个异步爬虫，支持并发限制、重试和断点续传，并解释关键设计。',
 '解释 Linux 下 NUMA 架构对大模型 CPU 推理的影响，并给出调优建议。',
 'Write a Rust function that parses a CSV file into structs with serde, with error handling and tests.',
 '帮我审查这段代码的问题并给出修改：\n```js\nasync function load(ids){ let r=[]; ids.forEach(async id=>{ r.push(await fetch("/api/"+id)) }); return r }\n```',
 'Explain how speculative decoding works, including EAGLE and Medusa, and when it fails to speed up inference.',
 '写一份 systemd 服务配置，运行一个 Python Web 服务，要求开机自启、失败重启、限制内存 4G，并解释每一行。',
 '你是一个编程助手。用户说：“把项目里所有 print 换成 logging，并保持输出级别一致”。请给出完整的执行计划和需要修改的代码示例。',
 'Given the tools [{"name":"read_file","parameters":{"path":"string"}},{"name":"run_shell","parameters":{"command":"string"}}], the user asks: find why the unit tests fail in ./tests. Reply with the tool calls you would make in JSON, step by step.',
 '证明：任意 n 个整数中，必有若干个数之和能被 n 整除。写出详细推导。',
 '把下面的需求整理成产品需求文档（PRD）：做一个家庭 NAS 上的照片自动整理工具，支持人脸聚类、按时间地点归档、重复照片检测。',
 'Write a bash script that monitors GPU memory with nvidia-smi every second and alerts when usage exceeds 90%, logging to a file with rotation.',
 '用 TypeScript + React 写一个可拖拽排序的待办列表组件，带本地存储，并解释状态管理。',
 '总结一下 Transformer 注意力机制的复杂度问题，以及 Linear Attention、稀疏注意力（如 DSA）各自如何解决。',
 'Translate to Chinese and polish: "The hybrid CPU/GPU MoE runtime keeps hot layers on the GPU while streaming expert weights from host memory, trading PCIe bandwidth for capacity."',
 '我的 Docker 容器启动后立刻退出，日志显示 exec format error，可能是什么原因？如何排查？',
 'Implement an LRU cache in C++17 with O(1) get/put, thread-safe, and write a short benchmark.',
 '写一个 SQL 查询：统计每个用户最近 30 天的订单数、总金额、最大单笔，并找出金额环比增长超过 50% 的用户。',
 '帮我写一封邮件给团队，说明下周服务器维护计划，包括时间、影响范围、回滚方案。',
 'Design a REST API for a multi-tenant note-taking app: endpoints, auth, pagination, error format. Give OpenAPI-style examples.',
 '用 Go 写一个 HTTP 反向代理，支持按路径前缀路由、健康检查和简单的限流。',
 '分析这段日志可能的问题：\n[ERROR] CUDA out of memory. Tried to allocate 800.00 MiB (GPU 0; 72.00 GiB total capacity; 70.10 GiB already allocated)\n给出排查步骤。',
 'Explain the difference between processes, threads and coroutines in Python, with examples using multiprocessing, threading and asyncio.',
 '为一个 Python 库写 README，包括安装、快速开始、配置项表格、常见问题。库的功能是把 Markdown 转成带目录的 HTML。',
 '如果一个数列满足 a1=1, a(n+1)=a(n)+2n+1，求通项公式并验证。然后写一个 Python 程序打印前 20 项。',
 'You are an agent with a shell. The task: set up a Python virtualenv, install requirements, run pytest and fix the first failure. Describe each command and what you expect to see.',
 '解释 git rebase 和 git merge 的区别，什么时候该用哪个，并给出交互式 rebase 整理提交历史的步骤。',
 'Write a Python script that reads a large JSONL file streaming, computes per-key statistics and writes a summary report in Markdown.',
 '请用通俗的语言给初中生讲解什么是神经网络，并举一个生活中的例子。',
 '给出一个基于 FastAPI 的文件上传服务，支持分片上传、校验 MD5、合并文件，附完整代码。',
 'Refactor this Python function for readability and performance:\n```python\ndef f(l):\n  r=[]\n  for i in range(len(l)):\n    if l[i] not in r: r.append(l[i])\n  return sorted(r, key=lambda x: -l.count(x))\n```',
 '写一个 Vue 3 组件：表格支持分页、排序、筛选、行内编辑，使用组合式 API。',
 'Compare PostgreSQL and MySQL for a write-heavy analytics workload; discuss indexing, partitioning, replication and tuning.',
]
with open(out, 'a', encoding='utf-8') as f:
    for i, p in enumerate(P):
        for temp in (0.0, 0.7):
            body = json.dumps(dict(model='GLM-5.3-Flash-NVFP4', messages=[{'role': 'user', 'content': p}],
                                   max_tokens=1200, temperature=temp)).encode()
            try:
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body,
                                             {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    m = json.loads(r.read())['choices'][0]['message']
                f.write((m.get('reasoning_content') or '') + '\n' + (m.get('content') or '') + '\n\n'); f.flush()
                print(i, temp, 'ok', flush=True)
            except Exception as exc:
                print(i, temp, 'fail', repr(exc)[:200], flush=True)
