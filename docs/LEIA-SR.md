# Leia SR output: hardware checklist

For an owner of a glasses-free, eye-tracked Leia SR display (Acer SpatialLabs, Samsung Odyssey 3D, Lenovo ThinkPad 3D
and others). The whole chain is implemented and checked without hardware; only the actual weave on a panel still
needs checking with real hardware.

## How it works

- The game renders each eye with its own camera (native geometric stereo, the same as the other 3D modes).
- `stereo = leia` composes the two eyes as a side-by-side pair of two views at the size the SR SDK recommends
  (`getRecommendedViewsTextureWidth/Height`, read as in the SDK's `directx12_weaving` example). The SDK's default is
  half the panel's physical resolution in each direction: 1920 × 1080 per view on a 4K panel, a 3840 × 1080 pair. If
  the SDK gives no size, the game uses half the window. The woven image keeps the window's size.
- With **Resolution** on Auto, the Leia SR output renders each eye at 1080p, the size of one view on 4K SR panels.
  Until 2026-11-19 each eye was rendered at the screen's resolution (2160p on a 4K panel) and the pair was
  2 × 3840 × 2160: four times the pixels the weaver uses.
- `sar_leia_sr.dll`, next to `recomp_main.exe`, is built with MSVC against Leia's SR SDK through SR-lib
  (`recomp/leia/construire.py`). It:
  - opens a Direct3D 12 device on the same graphics card as the game's Vulkan renderer;
  - shares the pair, the woven image and a fence with Vulkan;
  - weaves the pair with the SDK's DX12 weaver, using the call pattern verified on SR panels in the author's other
    ports: sRGB conversion off, viewport and scissor set on the command list right before `Weave()`, the input
    re-set every frame, and SR-lib's defaults of one frame of latency and late latching.
- The game copies the woven image to the window and presents it.
- `recomp_main.exe` has no SR imports at all. The module loads every SR DLL with delay loading. Without the module,
  the SR Platform or an SR display, the pair is shown side by side and the game logs why.

## Prerequisites

- Windows 10 or 11, x64.
- The **SR Platform** installed, with its service running. It provides the `SimulatedReality*` and `DimencoWeaving`
  DLLs; SpaceAnimalsRecomp ships none of them.
- The SR display connected and enabled in Windows.
- A graphics driver with Vulkan `VK_KHR_external_memory_win32` and `VK_KHR_external_semaphore_win32`. NVIDIA, AMD and
  Intel drivers have them.
- The Visual C++ 2015–2022 x64 runtime, which the SR Platform also needs.
- `sar_leia_sr.dll` in `recomp\build\full\`, next to `recomp_main.exe`.

## Launch

1. Launcher, **3D** page: set **3D mode** to **Leia SR (eye tracking)** and **Swap eyes** to **No**. With the eyes
   swapped, the weaver gets the right eye on the left and depth comes out inside out.
2. **Graphics**: use borderless full screen. The launcher shows an amber "use full screen" warning in windowed mode.
3. Put the game window on the SR display. The weaver reads the window's position on the panel.
4. Play. The log is at `recomp\build\full\recomp_run.log`, or wherever `--log` points.

## Checklist

| # | Check | Expected | If it fails |
|---|---|---|---|
| 1 | Weaver starts | Log: `[LeiaSR] weaver initialized.` then `Leia SR: weaver up` | `CreateSRInterfaceDX12 hr 0x80004002`: no SR Platform, service or display found. `sar_leia_sr.dll missing`: the module or the VC++ runtime is missing. |
| 2 | Sizes | `[LeiaSR] display: ... recommended view VW x VH.` (1920x1080 expected on a 4K panel), then `Leia SR: pair 2VW x VH (view VW x VH, recommended by the SR SDK), woven image W x H` with W × H the panel's resolution (borderless full screen) | A smaller W × H means the window isn't full screen on the SR display. "half the window" instead of "recommended by the SR SDK": the SDK gave no size; send the log. |
| 3 | First weave | `[LeiaSR] weaving W x H (pair 2VW x VH).` (no "simulated") | `the weave failed` or `device removed`: send the log. To compare with the old full-size pair: `RECOMP_LEIA_VIEW=full`. |
| 4 | Gamma | Colours as in 2D mode, not washed out or too dark | Report it. The pair is UNORM with gamma already encoded, and the weaver's sRGB conversion is off as in the verified ports. |
| 5 | Depth direction | The Arwing stands out in front of the far background | Set **Swap eyes** to **No**. |
| 6 | Head tracking | Moving the head shifts parallax smoothly, without jitter | Send the log and a phone video if possible. |
| 7 | HUD | Readable in both eyes, at the interface depth set on the 3D page | Try an interface depth of 0. |
| 8 | Frame rate | About 60 fps (log: `[gfx] 60.0 fps`) | The weave adds one Direct3D 12 pass per frame. Report the GPU and the fps. |
| 9 | Exit | Closing the game puts the panel back in 2D (log: `Leia SR: weaver released`) | Report it. |

## Diagnostics without hardware

- `SAR_LEIA_SIMULATE=1`: create succeeds without the SR runtime, and the weave becomes a known copy: the left half of
  the left eye beside the right half of the right eye.
- `SAR_LEIA_SIMULATE_VIEW=<W>x<H>`: the view size the simulated SDK recommends (none by default). When the pair is
  smaller than the window, the simulated weave copies it unscaled into the top-left corner and leaves the rest black.
- `RECOMP_LEIA_VIEW=full` (the old pair of two window-sized views) or `RECOMP_LEIA_VIEW=<W>x<H>`: overrides the view
  size.
- `RECOMP_LEIA_DUMP=<frame>`: writes `leia_pair_<frame>.png` and `leia_woven_<frame>.png` to `RECOMP_SNAPSHOT_DIR`,
  read back after the weave.
- Checked on an RTX 3080 (2026-11-16):
  - the simulated weave is bit-exact against the pair (max difference 0, both halves);
  - the pair has the same gamma as the normal presentation;
  - 60 fps in simultaneous stereo;
  - without the SR runtime: `hr 0x80004002`, then side by side at 60 fps;
  - without the module: side by side at 60 fps.
- Checked on the same card (2026-11-19), 1920 × 1080 window: with no recommended size, pair 1920 × 540 (half the
  window); with `SAR_LEIA_SIMULATE_VIEW=1280x720`, pair 2560 × 720. In both, the woven image matches the pair exactly
  (rest black), and each half of the pair is the whole eye scaled down (correlation 0.998 with that eye's snapshot).
- `SAR_LEIA_DEBUG=1` turns on the Direct3D 12 debug layer in the module, if Windows' Graphics Tools are installed.

## Building the module

```
python recomp/leia/construire.py --sr-lib <SR-lib checkout>
```

- Needs the MSVC x64 build tools, found through vswhere.
- SR-lib is the MIT wrapper at github.com/bo3b/SR-lib, containing `SR-SDK-<version>/x64`. Point to it with
  `--sr-lib`, `SAR_SR_LIB_ROOT`, or put it at `recomp/leia/SR-lib` (ignored by git).
- The proprietary SDK is never committed.
- `--sans-sr` builds a simulate-only module.
- The output goes to `recomp/build/full/sar_leia_sr.dll`.
