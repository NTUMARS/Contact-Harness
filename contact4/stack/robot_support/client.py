"""Request/reply client of the robot server (msgpack over a ZeroMQ REQ socket)."""
import msgpack
import zmq


class RobotClientError(Exception):
    """The robot server refused a request, or no reply arrived in time."""


class RobotClient:
    def __init__(self, address, timeout=1.0):
        self.address = address
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:  # noqa: BLE001
                pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self._set_timeout(self.timeout)
        self.socket.connect(self.address)

    def _set_timeout(self, seconds):
        self.socket.setsockopt(zmq.RCVTIMEO, int(seconds * 1000))
        self.socket.setsockopt(zmq.SNDTIMEO, int(seconds * 1000))

    def request(self, method, params=None, timeout=None):
        """Send one request and return the result field of the reply."""
        if timeout is not None:
            self._set_timeout(timeout)
        try:
            self.socket.send(msgpack.packb({"method": method, "params": {} if params is None else params},
                                           use_bin_type=True))
            reply = msgpack.unpackb(self.socket.recv(), raw=False)
            if not reply.get("success", False):
                raise RobotClientError("%s failed: %s" % (method, reply.get("error", "Unknown error")))
            return reply.get("result")
        except zmq.Again:
            # A REQ socket that sent without receiving cannot send again; start from a fresh socket.
            self._connect()
            raise RobotClientError("%s timeout after %ss" % (method, self.timeout if timeout is None else timeout))
        except zmq.ZMQError as exc:
            self._connect()
            raise RobotClientError("%s failed: %s" % (method, exc)) from exc
        except RobotClientError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RobotClientError("%s failed: %s" % (method, exc)) from exc
        finally:
            if timeout is not None and self.socket is not None:
                self._set_timeout(self.timeout)

    def get_robot_info(self):
        result = self.request("get_robot_info")
        if result is None:
            raise RobotClientError("get_robot_info returned None")
        return result

    def get_tool_info(self):
        result = self.request("get_tool_info")
        if result is None:
            raise RobotClientError("get_tool_info returned None")
        return result

    def close(self):
        if self.socket is not None:
            self.socket.close()
        self.context.term()
