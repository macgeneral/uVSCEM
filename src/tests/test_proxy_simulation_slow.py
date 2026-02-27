from __future__ import annotations

import asyncio
import http.server
import json
import select
import socket
import socketserver
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, cast

import pytest

from uvscem.api_client import CodeAPIManager
from uvscem.extension_manager import CodeExtensionManager
from uvscem.vsce_sign_bootstrap import provision_vsce_sign_binary_for_run

pytestmark = pytest.mark.slow


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class _NaiveForwardProxyHandler(http.server.BaseHTTPRequestHandler):
    timeout = 10

    def do_CONNECT(self) -> None:
        threading.current_thread()._uvscem_proxy_handler = True  # type: ignore[attr-defined]
        target_host, _, target_port_raw = self.path.partition(":")
        target_port = int(target_port_raw) if target_port_raw else 443

        with socket.create_connection(
            (target_host, target_port), timeout=self.timeout
        ) as upstream:
            self.send_response(200, "Connection Established")
            self.end_headers()

            sockets = [self.connection, upstream]
            while True:
                readable, _, _ = select.select(sockets, [], [], 0.5)
                if not readable:
                    continue

                if self.connection in readable:
                    data = self.connection.recv(65536)
                    if not data:
                        break
                    upstream.sendall(data)

                if upstream in readable:
                    data = upstream.recv(65536)
                    if not data:
                        break
                    self.connection.sendall(data)

    def log_message(self, format: str, *args) -> None:
        return None


@contextmanager
def _run_naive_proxy() -> Iterator[tuple[str, int]]:
    with _ThreadingTCPServer(("127.0.0.1", 0), _NaiveForwardProxyHandler) as server:
        host, port = cast(tuple[str, int], server.server_address)
        server_thread = threading.Thread(
            target=server.serve_forever,
            name="proxy-server",
            daemon=True,
        )
        server_thread.start()
        try:
            yield host, port
        finally:
            server.shutdown()
            server_thread.join(timeout=3)


def _is_proxy_thread() -> bool:
    current = threading.current_thread()
    return current.name.startswith("proxy-server") or bool(
        getattr(current, "_uvscem_proxy_handler", False)
    )


@pytest.mark.slow
def test_proxy_only_simulation_blocks_direct_and_allows_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _run_naive_proxy() as (proxy_host, proxy_port):
        proxy_url = f"http://{proxy_host}:{proxy_port}"

        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.setenv("https_proxy", proxy_url)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")

        original_create_connection = socket.create_connection

        def _guarded_create_connection(address, *args, **kwargs):
            host, port = address

            if _is_proxy_thread():
                return original_create_connection(address, *args, **kwargs)

            if host in {"127.0.0.1", "localhost"} and int(port) == proxy_port:
                return original_create_connection(address, *args, **kwargs)

            raise RuntimeError(
                f"Direct egress blocked in proxy simulation: {host}:{port}"
            )

        monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)

        with pytest.raises(RuntimeError, match="Direct egress blocked"):
            socket.create_connection(("marketplace.visualstudio.com", 443), timeout=5)

        manager = CodeAPIManager()
        metadata = asyncio.run(manager.get_extension_metadata("dbaeumer.vscode-eslint"))
        assert "dbaeumer.vscode-eslint" in metadata
        assert metadata["dbaeumer.vscode-eslint"]


@pytest.mark.slow
def test_proxy_only_simulation_full_extension_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _run_naive_proxy() as (proxy_host, proxy_port):
        proxy_url = f"http://{proxy_host}:{proxy_port}"

        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.setenv("https_proxy", proxy_url)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")

        original_create_connection = socket.create_connection

        def _guarded_create_connection(address, *args, **kwargs):
            host, port = address

            if _is_proxy_thread():
                return original_create_connection(address, *args, **kwargs)

            if host in {"127.0.0.1", "localhost"} and int(port) == proxy_port:
                return original_create_connection(address, *args, **kwargs)

            raise RuntimeError(
                f"Direct egress blocked in proxy simulation: {host}:{port}"
            )

        monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)

        extension_id = "dbaeumer.vscode-eslint"
        config_path = tmp_path / "devcontainer.json"
        config_path.write_text(
            json.dumps(
                {
                    "customizations": {
                        "vscode": {"extensions": [extension_id]},
                    }
                }
            ),
            encoding="utf-8",
        )

        manager = CodeExtensionManager(
            config_name=str(config_path),
            code_path="code",
            target_directory=str(tmp_path / "cache"),
        )

        metadata_map = asyncio.run(
            manager.api_manager.get_extension_metadata(extension_id)
        )
        versions = metadata_map.get(extension_id, [])
        if not versions:
            pytest.skip("Could not resolve extension metadata for proxy simulation")

        manager.extension_metadata[extension_id] = versions[0]

        vsix_path = asyncio.run(manager.download_extension(extension_id))
        sig_path = asyncio.run(manager.download_signature_archive(extension_id))

        assert vsix_path.is_file()
        assert sig_path.is_file()
        assert vsix_path.stat().st_size > 0
        assert sig_path.stat().st_size > 0

        with provision_vsce_sign_binary_for_run(install_dir=tmp_path / "bin") as binary:
            manager.vsce_sign_binary = binary
            try:
                asyncio.run(manager.verify_extension_signature(vsix_path, sig_path))
            finally:
                manager.vsce_sign_binary = None
