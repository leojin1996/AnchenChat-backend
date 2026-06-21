# AnchenChat Backend

AnchenChat 后端服务，基于 FastAPI，提供聊天、语音转写、语音合成、登录鉴权、销售问答和联网搜索等接口。

本仓库已包含 Dockerfile、Docker Compose、Nginx 反向代理配置和部署脚本，推荐在云服务器上使用 Docker Compose 部署。

## 运行环境

云服务器需要提前安装：

- Docker
- Docker Compose 插件，或旧版 `docker-compose`
- Git
- curl

推荐服务器至少开放以下端口：

- `22`：SSH 登录
- `80`：HTTP 访问，默认由 Nginx 暴露
- `443`：HTTPS，建议由云厂商负载均衡、证书服务或外层 Nginx/Caddy 负责

应用容器内部监听 `8000`，不建议直接暴露到公网。

## 首次部署

登录云服务器后克隆仓库：

```bash
git clone https://github.com/leojin1996/AnchenChat-backend.git
cd AnchenChat-backend
```

创建生产环境变量文件：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置以下生产值：

- `OPENAI_API_KEY`：OpenAI 或兼容服务的 API Key
- `OPENAI_BASE_URL`：OpenAI 兼容接口地址，默认 `https://api.openai.com/v1`
- `OPENAI_CHAT_MODEL`：聊天模型
- `AUTH_JWT_SECRET`：JWT 签名密钥，生产环境必须使用 32 字节以上随机字符串
- `BACKEND_CORS_ORIGINS`：前端域名，生产环境建议不要使用 `*`
- `HTTP_PORT`：Nginx 对外端口，默认 `80`

可用下面命令生成 JWT 密钥：

```bash
openssl rand -hex 32
```

如果启用手机号验证码登录，继续配置：

- `AUTH_ENABLED=true`
- `AUTH_ALLOWLIST_PATH=auth/allowlist.yaml`
- `ALIYUN_SMS_ACCESS_KEY_ID`
- `ALIYUN_SMS_ACCESS_KEY_SECRET`
- `ALIYUN_SMS_SIGN_NAME`
- `ALIYUN_SMS_TEMPLATE_CODE`

生产环境不要配置 `AUTH_DEV_BYPASS_CODE`。

创建登录白名单：

```bash
cp auth/allowlist.example.yaml auth/allowlist.yaml
```

编辑 `auth/allowlist.yaml`，填入允许登录的手机号和名称：

```yaml
users:
  - phone: "13800138000"
    name: "管理员"
    role: "admin"
```

如果需要销售问答功能，还需要在 `.env` 中配置 SQL Server：

- `SQL_SERVER_HOST_NAME`
- `SQL_SERVER_USER_NAME`
- `SQL_SERVER_USER_PASSWORD`
- `SQL_SERVER_DATABASE`
- `SQL_SERVER_SCHEMA`
- `SQL_SERVER_QUERY_TIMEOUT`

如果需要联网搜索功能，配置：

- `TAVILY_API_KEY`
- `TAVILY_BASE_URL`
- `TAVILY_SEARCH_DEPTH`
- `TAVILY_MAX_RESULTS`

启动服务：

```bash
chmod +x deploy.sh
./deploy.sh up
```

部署脚本会自动构建镜像、启动后端和 Nginx，并检查：

```bash
curl http://127.0.0.1:${HTTP_PORT:-80}/health
```

返回以下内容表示服务可用：

```json
{"status":"ok"}
```

## 常用运维命令

查看服务状态：

```bash
./deploy.sh status
```

查看日志：

```bash
./deploy.sh logs
```

重启并重新构建：

```bash
./deploy.sh restart
```

检查健康状态：

```bash
./deploy.sh health
```

停止服务：

```bash
./deploy.sh stop
```

更新线上代码：

```bash
git pull
./deploy.sh restart
```

## HTTPS 和域名

当前仓库内置的 Nginx 只监听容器网络里的 HTTP。

生产环境建议使用以下任一方式提供 HTTPS：

- 云厂商负载均衡绑定证书，并将流量转发到服务器 `HTTP_PORT`
- 在服务器外层部署 Caddy/Nginx/Traefik 负责 HTTPS 终止
- 使用云平台自带的容器 HTTPS 网关

如果前端部署在固定域名，请把 `.env` 中的 `BACKEND_CORS_ORIGINS` 改为前端地址，例如：

```env
BACKEND_CORS_ORIGINS=https://app.example.com
```

多个来源用英文逗号分隔：

```env
BACKEND_CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

## 容器平台部署

如果使用 Railway、Render、Fly.io、阿里云容器服务、腾讯云 CloudBase Run 等平台，可以直接使用 `Dockerfile` 构建。

容器启动命令已在 Dockerfile 中配置：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips '*'
```

容器平台需要设置：

- 构建上下文为仓库根目录
- 容器端口为 `8000`
- 健康检查路径为 `/health`
- 所有 `.env` 中的生产环境变量
- `auth/allowlist.yaml`，可通过平台的持久化文件、挂载卷或镜像内安全注入方式提供

## 安全注意事项

- 不要提交 `.env`。
- 不要提交真实的 `auth/allowlist.yaml`。
- `AUTH_JWT_SECRET` 必须足够长且只在服务器保存。
- 生产环境不要设置 `AUTH_DEV_BYPASS_CODE`。
- SQL Server 只允许后端服务器 IP 访问。
- 如果 `BACKEND_CORS_ORIGINS=*`，任意网页都能向后端发起跨域请求；生产环境建议改为明确的前端域名。

## 故障排查

查看最近日志：

```bash
docker compose logs --tail=200 backend nginx
```

如果启动失败，优先检查：

- `.env` 是否存在
- `auth/allowlist.yaml` 是否存在
- `AUTH_JWT_SECRET` 是否已配置
- OpenAI、Tavily、阿里云短信和 SQL Server 凭证是否正确
- 云服务器安全组是否开放 `HTTP_PORT`
- SQL Server 防火墙是否允许当前云服务器访问

如果健康检查失败，可以直接访问：

```bash
curl -i http://127.0.0.1:${HTTP_PORT:-80}/health
```
