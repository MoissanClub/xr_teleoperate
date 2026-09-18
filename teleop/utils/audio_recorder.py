"""G1 microphone recorder for xr_teleoperate episodes.

Where the audio comes from
--------------------------
The G1 microphone array is wired to PC1 (192.168.123.161), not to PC2 or the host
running this script, so ALSA/PyAudio cannot see it. PC1 continuously publishes it on the
wired robot network as UDP multicast 239.168.123.161:5555: raw PCM, 16 kHz, 16-bit
little-endian, mono, in 5120-byte datagrams (2560 samples = 160 ms). The datagrams carry
no header and no timestamps. This matches `info.audio` in EpisodeWriter's data.json.

(Standalone equivalent used to prototype this: unitree_g1_dev/pc2/audio/record_g1_mic.py.)

How it fits into an episode
---------------------------
    receiver thread ──> in-RAM sample buffer (only while an episode is running)
    EpisodeWriter.create_episode()  -> begin_episode()   start buffering
    EpisodeWriter.add_item()        -> stamps time.monotonic() per frame (writer side)
    EpisodeWriter._save_episode()   -> finish_episode()  cut + write files

Frames are not perfectly periodic (IK time varies) and datagrams arrive in 160 ms bursts, so
audio is NOT sliced at a fixed 533.33 samples per frame. Instead each frame owns the audio
between its own timestamp and the next frame's timestamp, which keeps audio and video on the
same wall clock. Consecutive windows share boundaries, so there are no gaps or overlaps.

Outputs, per episode (inside `audios/`):
    audio_{idx:06d}_mic_0.npy   int16 array, one per frame (Unitree's per-frame format)
    audio.wav                   the per-frame windows concatenated, 16 kHz mono 16-bit, so
                                WAV time == frame_idx / fps and it plays in sync with video
"""
import os
import socket
import struct
import threading
import time
import wave

import numpy as np
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

DEFAULT_GROUP = "239.168.123.161"     # multicast group PC1 publishes the mic on
DEFAULT_PORT = 5555
DEFAULT_ROBOT_IP = "192.168.123.161"  # PC1; only used to pick the right local NIC
SAMPLE_RATE = 16000
MIC_NAME = "mic_0"                    # key used in data.json "audios" and in file names

# A datagram can be up to 65535 bytes and recvfrom(n) silently truncates anything longer than
# n. The stream's 5120-byte datagrams are bigger than the usual 4096 default, so always read
# the maximum.
MAX_DATAGRAM = 65535

FIRST_PACKET_TIMEOUT_S = 5.0

# Stream sample 0 of an episode is assumed to have been captured AUDIO_LATENCY_S seconds before
# its datagram arrived (a datagram holds up to 160 ms of audio, and capture/network add more).
# A larger value moves audio events earlier in the episode relative to the video. Calibrate it
# with a clap test in view of the head camera; 0.0 = treat arrival time as capture time.
AUDIO_LATENCY_S = 0.0

# How long finish_episode() waits for the datagram(s) that contain the last frames' audio.
TAIL_WAIT_S = 0.4


def local_ip_towards(remote_ip):
    """Local IPv4 address the kernel would use to reach `remote_ip`.

    Multicast membership is per network interface, and this machine may have several
    (wired robot LAN, Wi-Fi). Connecting a UDP socket sends nothing; it only runs the routing
    lookup, and getsockname() then reveals which local address/interface was chosen.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((remote_ip, 9))
        return probe.getsockname()[0]
    finally:
        probe.close()


class G1MicRecorder:
    def __init__(self, group=DEFAULT_GROUP, port=DEFAULT_PORT, robot_ip=DEFAULT_ROBOT_IP,
                 latency_s=AUDIO_LATENCY_S):
        self.group = group
        self.port = port
        self.robot_ip = robot_ip
        self.latency_s = latency_s

        self._sock = None
        self._thread = None
        self._stop = threading.Event()
        self._first_packet = threading.Event()
        self._lock = threading.Lock()   # guards the fields below, shared with the receiver thread

        self._recording = False
        self._chunks = []               # int16 arrays received during the current episode
        self._t0 = None                 # monotonic arrival time of the episode's first datagram
        self._n_samples = 0
        self._peak_seen = 0             # loudest |sample| ever received (silence detection)
        self._error = None              # receiver thread failure, if any

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        """Join the multicast group and start receiving. Raises RuntimeError if nothing arrives."""
        iface_ip = local_ip_towards(self.robot_ip)
        logger_mp.info(f"==> G1MicRecorder joining {self.group}:{self.port} on interface {iface_ip}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # allow other listeners
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind((self.group, self.port))                            # only this group's packets
        membership = struct.pack("4s4s", socket.inet_aton(self.group), socket.inet_aton(iface_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sock.settimeout(0.5)                                          # lets the loop see _stop
        self._sock = sock

        self._thread = threading.Thread(target=self._receive_loop, name="g1-mic", daemon=True)
        self._thread.start()

        if not self._first_packet.wait(FIRST_PACKET_TIMEOUT_S):
            self.stop()
            raise RuntimeError(
                f"No microphone audio received from PC1 within {FIRST_PACKET_TIMEOUT_S:.0f} s "
                f"(multicast {self.group}:{self.port} on {iface_ip}). Check that {self.robot_ip} "
                "is reachable and that no firewall blocks multicast.")

        # Give a moment of stream so an all-zero stream (e.g. the robot's Voice Assistant in
        # "Closed mode", reported in unitree_sdk2_python issue #143) is noticed at startup.
        time.sleep(0.5)
        if self._peak_seen == 0:
            logger_mp.warning("G1 microphone stream is connected but carries only zeros. "
                              "Episodes will be silent; check the robot's voice assistant mode.")
        logger_mp.info("==> G1MicRecorder receiving audio.")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _receive_loop(self):
        carry = b""   # a leftover odd byte, so samples stay 2-byte aligned across datagrams
        while not self._stop.is_set():
            try:
                datagram = self._sock.recvfrom(MAX_DATAGRAM)[0]
            except socket.timeout:
                continue
            except OSError as e:
                self._error = e
                logger_mp.error(f"G1MicRecorder receive failed: {e}")
                return
            arrival = time.monotonic()

            data = carry + datagram
            carry = data[-1:] if len(data) % 2 else b""
            if carry:
                data = data[:-1]
            samples = np.frombuffer(data, dtype="<i2")
            if samples.size == 0:
                continue

            peak = int(np.abs(samples.astype(np.int32)).max())
            with self._lock:
                self._peak_seen = max(self._peak_seen, peak)
                if self._recording:
                    if self._t0 is None:
                        self._t0 = arrival
                    self._chunks.append(samples)
                    self._n_samples += samples.size
            self._first_packet.set()

    # ------------------------------------------------------------------ episodes
    def begin_episode(self):
        """Start buffering audio for a new episode (drops anything left from a previous one)."""
        with self._lock:
            self._chunks = []
            self._n_samples = 0
            self._t0 = None
            self._recording = True

    def finish_episode(self, frame_times, frequency, audio_dir, mic=MIC_NAME):
        """Cut the buffered audio into per-frame windows and write the episode's audio files.

        frame_times: time.monotonic() stamp of every recorded frame, in frame order.
        frequency:   recording rate in Hz (only used to size the last frame's window).
        Returns the number of frames written.
        """
        n_frames = len(frame_times)
        if n_frames == 0:
            with self._lock:
                self._recording = False
            return 0

        # Datagrams arrive up to ~160 ms after their audio was captured, so the last frames'
        # audio may not have landed yet. Wait briefly for it before closing the buffer.
        t_end = frame_times[-1] + 1.0 / frequency
        deadline = time.monotonic() + TAIL_WAIT_S
        while time.monotonic() < deadline:
            with self._lock:
                t0, n = self._t0, self._n_samples
            if t0 is not None and (t0 - self.latency_s) + n / SAMPLE_RATE >= t_end:
                break
            time.sleep(0.02)

        with self._lock:
            self._recording = False
            chunks, t0 = self._chunks, self._t0
            self._chunks = []
        stream = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

        if t0 is None:
            logger_mp.error("No microphone audio was received during this episode; "
                            "audio files will be silent.")
            t0 = frame_times[0]
        anchor = t0 - self.latency_s   # wall-clock time that stream sample 0 corresponds to

        # Sample boundaries: frame k covers [bounds[k], bounds[k+1]).
        times = list(frame_times) + [t_end]
        bounds = [int(round((t - anchor) * SAMPLE_RATE)) for t in times]

        os.makedirs(audio_dir, exist_ok=True)
        windows = []
        padded = 0
        for k in range(n_frames):
            a, b = bounds[k], max(bounds[k + 1], bounds[k])
            window = np.zeros(b - a, dtype=np.int16)   # zeros where no audio exists
            lo, hi = max(a, 0), min(b, stream.size)
            if hi > lo:
                window[lo - a:hi - a] = stream[lo:hi]
            padded += (b - a) - max(hi - lo, 0)
            np.save(os.path.join(audio_dir, f"audio_{str(k).zfill(6)}_{mic}.npy"), window)
            windows.append(window)

        full = np.concatenate(windows)
        with wave.open(os.path.join(audio_dir, "audio.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(full.astype("<i2").tobytes())

        if padded > 0.1 * SAMPLE_RATE:
            logger_mp.warning(f"Audio is missing {padded / SAMPLE_RATE:.2f} s of the "
                              f"{full.size / SAMPLE_RATE:.2f} s episode (zero-padded); "
                              "packets were dropped or the stream started late.")
        if full.size and not full.any():
            logger_mp.warning("This episode's audio is entirely silent (all zeros).")
        logger_mp.info(f"==> Audio saved: {n_frames} frames, {full.size / SAMPLE_RATE:.2f} s -> {audio_dir}")
        return n_frames
