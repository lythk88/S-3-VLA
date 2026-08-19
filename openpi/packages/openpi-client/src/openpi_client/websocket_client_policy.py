import logging
import inspect
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connect_kwargs = {
                    "compression": None,
                    "max_size": None,
                    "additional_headers": headers,
                }
                # websockets >=15 exposes keepalive on the sync client, while
                # websockets 13 forwards unknown kwargs to socket.create_connection.
                # Disable keepalive only when the installed API supports it; the
                # older sync client has no keepalive thread to disable.
                if "ping_interval" in inspect.signature(
                    websockets.sync.client.connect
                ).parameters:
                    connect_kwargs["ping_interval"] = None
                conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict, noise=None) -> Dict:  # noqa: UP006
        payload = dict(obs)
        if noise is not None:
            payload["__debug_noise__"] = noise
        data = self._packer.pack(payload)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
