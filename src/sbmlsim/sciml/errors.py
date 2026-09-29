"""Errors of the neural networks."""

from __future__ import annotations


class NetworkImportError(ValueError):
    """A network or its arrays cannot be read.

    The message names the network and, where it applies, the layer and the
    array.
    """


class UnsupportedLayerError(NotImplementedError):
    """A node of the forward pass has no implementation.

    Attributes:
        network: id of the network.
        node: name of the node of the forward pass.
        target: the layer type, function or method of the node.
        reason: why the node is not evaluated.
    """

    def __init__(self, network: str, node: str, target: str, reason: str) -> None:
        """Initialize the error.

        Args:
            network: id of the network.
            node: name of the node of the forward pass.
            target: the layer type, function or method of the node.
            reason: why the node is not evaluated.
        """
        self.network = network
        self.node = node
        self.target = target
        self.reason = reason
        super().__init__(
            f"Network '{network}', node '{node}': '{target}' is not supported, {reason}"
        )
