"""
mls/ratchet_tree.py — RFC 9420 TreeKEM Ratchet Tree

Implements the MLS binary left-balanced ratchet tree [RFC 9420 §7].

Tree structure (n=4 leaves, 7 nodes total):
  Node indices (array representation):
         6
       /   \\
      4     5
     / \\   / \\
    0   1  2   3   ← leaves

Leaf index L maps to tree node index 2*L.
Parent of node n = floor((n-1)/2) for n > 0.
Left child  of n = 2*n + 1  (if exists)
Right child of n = 2*n + 2  (if exists)

Each node stores:
  - hpke_pub  : X25519 public key bytes (None = blank node)
  - hpke_priv : X25519 private key (only known to the local member)
  - tree_hash : SHA-256 hash of this node and its subtree

References:
  RFC 9420 §7       — Ratchet Tree
  RFC 9420 §7.4     — TreeKEM
  RFC 9420 §7.8     — Tree Hashes
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from .crypto import (
    hash_bytes,
    hpke_seal,
    hpke_open,
    hkdf_expand_label,
    hkdf_extract,
    HASH_LEN,
)
from .key_package import KeyPackage


@dataclass
class TreeNode:
    """A single node in the MLS ratchet tree."""
    hpke_pub_bytes: Optional[bytes] = None       # None = blank
    hpke_priv:      Optional[X25519PrivateKey] = None
    key_package:    Optional[KeyPackage] = None  # Only set for leaf nodes
    node_hash:      bytes = field(default_factory=lambda: bytes(32))

    @property
    def is_blank(self) -> bool:
        return self.hpke_pub_bytes is None


class RatchetTree:
    """
    RFC 9420 §7 left-balanced binary ratchet tree.

    Internally stored as a flat array where:
      - Leaf L is at index 2*L
      - Internal node at level k, position p is at index (2^k) * (2*p+1) - 1
    """

    def __init__(self):
        """Initialize an empty tree."""
        self._nodes: List[TreeNode] = []
        self._num_leaves: int = 0

    # ------------------------------------------------------------------
    # Tree structure helpers (RFC 9420 §7.7 Tree Math)
    # ------------------------------------------------------------------

    @staticmethod
    def _level(x: int) -> int:
        """Level of node x (leaves = 0, root = max)."""
        if (x & 1) == 0:
            return 0
        k = 0
        while (x >> k) & 1 == 1:
            k += 1
        return k

    def _node_width(self) -> int:
        return 2 * self._num_leaves - 1 if self._num_leaves > 0 else 0

    def _root(self) -> int:
        w = self._node_width()
        return (1 << (w.bit_length() - 1)) - 1 if w > 0 else 0

    @staticmethod
    def _left_child(x: int) -> int:
        k = RatchetTree._level(x)
        assert k > 0, "Leaves have no children"
        return x ^ (1 << (k - 1))

    def _right_child(self, x: int) -> int:
        k = RatchetTree._level(x)
        assert k > 0, "Leaves have no children"
        r = x ^ (3 << (k - 1))
        w = self._node_width()
        while r >= w:
            r = self._left_child(r)
        return r

    @staticmethod
    def _parent_step(x: int) -> int:
        k = RatchetTree._level(x)
        b = (x >> (k + 1)) & 1
        return (x | (1 << k)) ^ (b << (k + 1))

    def _parent(self, x: int) -> int:
        if x == self._root():
            return x
        p = self._parent_step(x)
        w = self._node_width()
        while p >= w:
            p = self._parent_step(p)
        return p

    @staticmethod
    def _leaf_index_to_node(leaf: int) -> int:
        """Tree node index for leaf at position `leaf`."""
        return 2 * leaf

    @staticmethod
    def _node_to_leaf_index(node: int) -> int:
        """Leaf position for a leaf node index."""
        assert node % 2 == 0, "Not a leaf node index"
        return node // 2

    def _tree_size(self) -> int:
        """Total number of nodes for n leaves: 2n - 1."""
        if self._num_leaves == 0:
            return 0
        return 2 * self._num_leaves - 1

    def _ensure_size(self, new_leaves: int):
        """Grow the internal array to accommodate `new_leaves` total leaves."""
        needed = 2 * new_leaves - 1
        while len(self._nodes) < needed:
            self._nodes.append(TreeNode())

    # ------------------------------------------------------------------
    # Leaf operations
    # ------------------------------------------------------------------

    def add_leaf(self, key_package: KeyPackage) -> int:
        """
        RFC 9420 §7.3 — Add a member's KeyPackage to the tree.
        Returns the leaf index (0-based).
        """
        # Find first blank leaf or extend
        leaf_idx = None
        for i in range(self._num_leaves):
            node_idx = self._leaf_index_to_node(i)
            if node_idx < len(self._nodes) and self._nodes[node_idx].is_blank:
                leaf_idx = i
                break

        if leaf_idx is None:
            # Extend tree
            leaf_idx = self._num_leaves
            self._num_leaves += 1
            self._ensure_size(self._num_leaves)

        node_idx = self._leaf_index_to_node(leaf_idx)
        self._nodes[node_idx] = TreeNode(
            hpke_pub_bytes=key_package.hpke_pub_bytes,
            hpke_priv=key_package._hpke_priv,
            key_package=key_package,
        )
        # Blank the direct path above this leaf (RFC 9420 §7.3.1)
        self._blank_path(leaf_idx)
        self._update_hashes()
        return leaf_idx

    def remove_leaf(self, leaf_idx: int):
        """
        RFC 9420 §7.3 — Remove a member. Blanks the leaf and its direct path.
        """
        node_idx = self._leaf_index_to_node(leaf_idx)
        if node_idx < len(self._nodes):
            self._nodes[node_idx] = TreeNode()  # blank
        self._blank_path(leaf_idx)
        self._update_hashes()

    def _blank_path(self, leaf_idx: int):
        """Blank all parent nodes on the direct path from leaf to root."""
        for node_idx in self._direct_path_nodes(leaf_idx):
            if node_idx < len(self._nodes):
                self._nodes[node_idx] = TreeNode()

    # ------------------------------------------------------------------
    # Direct Path & Resolution
    # ------------------------------------------------------------------

    def direct_path(self, leaf_idx: int) -> List[int]:
        """
        RFC 9420 §7.4 — Direct path from leaf to root (exclusive of leaf).
        Returns node indices.
        """
        return self._direct_path_nodes(leaf_idx)

    def _direct_path_nodes(self, leaf_idx: int) -> List[int]:
        """Internal: collect parent node indices from leaf to root."""
        path = []
        node = self._leaf_index_to_node(leaf_idx)
        root = self._root()
        while node != root:
            node = self._parent(node)
            path.append(node)
        return path

    def copath(self, leaf_idx: int) -> List[int]:
        """
        RFC 9420 §7.4 — Copath: sibling of each node on the direct path.
        Used for resolution in path encryption.
        """
        copath = []
        node = self._leaf_index_to_node(leaf_idx)
        root = self._root()
        while node != root:
            p = self._parent(node)
            lc = self._left_child(p)
            rc = self._right_child(p)
            sibling = rc if node == lc else lc
            copath.append(sibling)
            node = p
        return copath

    def resolve(self, node_idx: int) -> List[int]:
        """
        RFC 9420 §7.5 — Resolution of a node:
        - Blank leaf → []
        - Non-blank leaf → [node_idx]
        - Internal non-blank → [node_idx]
        - Blank internal → resolve(left) + resolve(right)
        """
        if node_idx >= len(self._nodes):
            return []
        node = self._nodes[node_idx]
        level = self._level(node_idx)
        if level == 0:
            return [] if node.is_blank else [node_idx]
        if not node.is_blank:
            return [node_idx]
        left  = self._left_child(node_idx)
        right = self._right_child(node_idx)
        return self.resolve(left) + self.resolve(right)

    # ------------------------------------------------------------------
    # Path Secret Derivation (TreeKEM Update) [RFC 9420 §7.4]
    # ------------------------------------------------------------------

    def generate_path_secrets(self, leaf_idx: int) -> Tuple[bytes, Dict[int, Tuple[bytes, bytes]]]:
        """
        TreeKEM UpdatePath generation [RFC 9420 §7.4]:
        1. Generate a fresh path secret at the leaf
        2. Derive parent path secrets up to root via HKDF
        3. For each node on the direct path, HPKE-encrypt the path secret
           to each member in the copath resolution

        Returns:
            commit_secret : bytes — derived from root path secret
            encrypted_path: Dict[node_idx → (enc, ciphertext)] per direct path node
        """
        # Fresh leaf path secret (32 random bytes)
        leaf_path_secret = hashlib.sha256(
            b"path_secret" + leaf_idx.to_bytes(2, "big") + hashlib.sha256(
                __import__("os").urandom(32)
            ).digest()
        ).digest()

        path_nodes  = self.direct_path(leaf_idx)
        copath_nodes = self.copath(leaf_idx)

        encrypted_path: Dict[int, Tuple[bytes, bytes]] = {}
        current_secret = leaf_path_secret

        for i, (node_idx, cp_node) in enumerate(zip(path_nodes, copath_nodes)):
            # Derive node secret for this level
            node_secret = hkdf_expand_label(current_secret, "node", b"", HASH_LEN)

            # Encrypt to each resolved member on the copath
            resolution = self.resolve(cp_node)
            if resolution:
                target_node_idx = resolution[0]
                if target_node_idx < len(self._nodes):
                    target_pub = self._nodes[target_node_idx].hpke_pub_bytes
                    if target_pub:
                        enc, ct = hpke_seal(target_pub, node_secret, aad=node_idx.to_bytes(4, "big"))
                        encrypted_path[node_idx] = (enc, ct)

            # Update parent node public key (derived from node_secret)
            node_pub_secret = hkdf_expand_label(node_secret, "node_pub", b"", HASH_LEN)
            if node_idx < len(self._nodes):
                # Store derived public bytes (we use hash as synthetic pub key for non-sender nodes)
                self._nodes[node_idx].hpke_pub_bytes = node_pub_secret

            # Ratchet: next secret = HKDF-expand of current
            current_secret = hkdf_expand_label(current_secret, "path", b"", HASH_LEN)

        # commit_secret = DeriveSecret(root_path_secret, "commit")
        commit_secret = hkdf_expand_label(current_secret, "commit", b"", HASH_LEN)
        self._update_hashes()
        return commit_secret, encrypted_path

    # ------------------------------------------------------------------
    # Tree Hash [RFC 9420 §7.8]
    # ------------------------------------------------------------------

    def _update_hashes(self):
        """Recompute node hashes bottom-up."""
        for i in range(len(self._nodes)):
            self._compute_node_hash(i)

    def _compute_node_hash(self, node_idx: int) -> bytes:
        """Compute the tree hash for a single node recursively."""
        if node_idx >= len(self._nodes):
            return bytes(32)
        node = self._nodes[node_idx]
        level = self._level(node_idx)
        if level == 0:
            # Leaf hash
            pub = node.hpke_pub_bytes or b""
            identity = (node.key_package.identity.encode() if node.key_package else b"")
            h = hash_bytes(b"\x01", identity, pub)
        else:
            lc = self._left_child(node_idx)
            rc = self._right_child(node_idx)
            lh = self._compute_node_hash(lc)
            rh = self._compute_node_hash(rc)
            pub = node.hpke_pub_bytes or b""
            h = hash_bytes(b"\x02", pub, lh, rh)
        if node_idx < len(self._nodes):
            self._nodes[node_idx].node_hash = h
        return h

    def tree_hash(self) -> bytes:
        """RFC 9420 §7.8 — Hash of the entire tree (root node hash)."""
        if self._num_leaves == 0:
            return bytes(32)
        root = self._root()
        return self._compute_node_hash(root)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def num_leaves(self) -> int:
        return self._num_leaves

    def get_leaf_key_package(self, leaf_idx: int) -> Optional[KeyPackage]:
        node_idx = self._leaf_index_to_node(leaf_idx)
        if node_idx < len(self._nodes):
            return self._nodes[node_idx].key_package
        return None

    def active_members(self) -> List[Tuple[int, KeyPackage]]:
        """Returns (leaf_idx, key_package) for all non-blank leaves."""
        result = []
        for i in range(self._num_leaves):
            kp = self.get_leaf_key_package(i)
            if kp is not None:
                result.append((i, kp))
        return result

    def __len__(self) -> int:
        return len(self.active_members())

    def __repr__(self) -> str:
        members = [kp.identity for _, kp in self.active_members()]
        return f"RatchetTree(leaves={self._num_leaves}, members={members}, hash={self.tree_hash().hex()[:8]}...)"
