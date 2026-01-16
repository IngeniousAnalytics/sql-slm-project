# Docker & WSL2 Setup (Postgres, MySQL, SQL Server)

This guide helps you run local database services inside Docker on Windows + WSL2 (Ubuntu) so the project can connect locally using the `.env` values.

## 1) Install Docker Desktop
- Install Docker Desktop for Windows and enable the WSL 2 backend.
- In Docker Desktop settings ➜ Resources ➜ WSL Integration, enable integration for your Ubuntu distro.
- Optional: increase memory and swap if you plan heavy local DB workloads.

## 2) Confirm Docker works in WSL2 Ubuntu
Open your Ubuntu WSL shell and run:

```bash
docker version
docker compose version
```

You should be able to run `docker` and `docker compose` directly from the WSL shell.

## 3) Configure `.env` values
Edit your project `.env` (do NOT commit it) with strong credentials. Keys used by `docker-compose.yml`:

- POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE
- MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE (MYSQL_ROOT_PASSWORD will use `MYSQL_PASSWORD` by default)
- SQLSERVER_PASSWORD (must meet MS SQL password complexity rules)

Examples:

```env
POSTGRES_USER=pguser
POSTGRES_PASSWORD=strong_pg_pass
POSTGRES_DATABASE=pg_db

MYSQL_USER=mysqluser
MYSQL_PASSWORD=strong_mysql_pass
MYSQL_DATABASE=mysql_db

SQLSERVER_PASSWORD=Str0ng!Passw0rd
```

Password note: SQL Server requires a complex SA password (uppercase, lowercase, number, symbol, min length 8).

## 4) Start databases
From the project root (where `docker-compose.yml` is located), run in WSL Ubuntu:

```bash
docker compose up -d
```

Check status:

```bash
docker compose ps
docker compose logs -f
```

## 5) Accessing databases
- PostgreSQL: host `localhost`, port `5432` (use `psql` or a GUI)
- MySQL: host `localhost`, port `3306` (use `mysql` or a GUI)
- SQL Server: host `localhost`, port `1433` (use `sqlcmd` or SSMS from Windows: `localhost,1433`)

From WSL the host is `localhost` as well when Docker Desktop WSL integration is enabled.

## 6) Stop and remove volumes (when needed)

```bash
docker compose down -v
```

## 7) Optional performance tips
- For heavier testing, raise CPU/Memory in Docker Desktop settings.
- On Windows, ensure virtualization is enabled in BIOS.

---
If you'd like, I can add a small init script to create demo schemas and sample data for each DB so you can run the project end-to-end quickly.