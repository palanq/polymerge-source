# polymerge — source mirror

polymerge is a Discord bot that merges several players' Battle of Polytopia screenshots of one map into a single composite showing everyone's explored territory combined.

This repository is a read-only mirror of the bot and merging script code. It is updated automatically on every change.

## What's here

| file | what it is |
|---|---|
| `polybot.py` | the Discord side: receives commands, downloads the screenshots, posts the result |
| `polymerge.py` | the image processing, run as a separate process by `polybot.py` |
| `requirements.txt` | the pinned dependencies |

This omits:
- Screenshots from real games.
- The board renders and sprites (source images used for merging).
- Scratch tooling, notes, development history.
