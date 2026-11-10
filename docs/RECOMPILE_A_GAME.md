# Recompile a Switch 2 game

One command does every step:

```
python tools\recompile_game.py "D:\path\to\your dump"
```

Use only games you own and dumped yourself. No games, keys or firmware are included.

## What the dump folder must look like

Either the extracted game itself:

```
My Game\
  exefs\   (main, main.npdm, ...)
  romfs\
```

or a game with its update (and DLC) next to it, as dump tools lay them out:

```
My Game\
  base\            exefs\, romfs\
  updates\458752\  exefs\, romfs\     (the newest update is used)
  dlc\...                             (found and reported; not added yet)
```

With an update, the update's code is recompiled, and its files go over the base's. The tool puts the two together in `<name>_dump` using hard links, so nothing large is copied.

## What it does

1. **Checks the dump** and explains how to extract it if `exefs\main` is missing.
2. **Checks the tools.** GCC is installed by `Install.bat`; the tool always uses GCC, never TCC. It also checks there is enough free disk space for the game's data plus about 25 GB to build.
3. **Builds the runtime** once, if it isn't built yet.
4. **Translates the game's code to C and compiles it.** A progress bar shows the time left. The first build of a big game takes 30–40 minutes; later builds only redo what changed.
5. **Turns on the native GPU path** (`gpu=nvn`) in the game's `.ini`.

The game ends up in `games\<name>\<name>.exe`, and everything the tool printed is also in `games\<name>\recompile.log`.

## Options

| Option | What it does |
|---|---|
| `--check` | Only checks that everything is ready; builds nothing. |
| `--name NAME` | Sets the game's name (default: taken from the dump folder). |
| `--out FOLDER` | Builds somewhere else. |
| `--replace` | Rebuilds over a game that is already in the output folder. Its saves and settings are kept. Close the game first. |
| `--jobs N` | Compilers run at a time. Use fewer if memory runs out. |

## The first time you play

Every new shader and texture is prepared the first time it appears, so new places stutter once. Later runs reuse what was prepared: the pipeline cache and the converted-texture pack in `astc_cache`.
