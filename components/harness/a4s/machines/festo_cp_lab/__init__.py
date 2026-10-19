"""The Festo CP Lab: one work center, two work cells, three OPC UA servers.

A model rather than a simulator. `machines/injection_molding` stands in for a
machine that is not there; this describes one that is, and everything the
harness does with it goes over the network to real hardware.
"""

from .model import BASE, SERVERS, SHELL_ID, build

__all__ = ["BASE", "SERVERS", "SHELL_ID", "build"]
