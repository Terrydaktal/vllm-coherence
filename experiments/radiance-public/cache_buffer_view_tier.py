"""Use verified buffer views for disk restores in the scheduler's snapshot tier."""

from cache_buffer_views import reader_variant, substitute
from qwen_radiance_chat_tier import ChatFileSystemTierManager, ChatStore


class BufferViewChatFileSystemTierManager(ChatFileSystemTierManager):
    def __init__(self, *, tier_type, **kwargs):
        # The factory uses the class name for discovery. Keep the existing
        # filesystem tier label in lookup events and cache metrics.
        super().__init__(tier_type="qwen_chat_fs", **kwargs)

    _read_many_views = staticmethod(reader_variant(ChatStore.read_many))
    _load = substitute(
        ChatFileSystemTierManager._load,
        'state["store"].read_many(disk_names, self._block_size)',
        'self._read_many_views(state["store"], disk_names, self._block_size)',
    )
