#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 agrofubris (module Leia SR, sous licence MIT : voir recomp/leia/sar_leia_sr.cpp).
"""Construit sar_leia_sr.dll (sortie Leia SR) avec MSVC : recomp/leia/sar_leia_sr.cpp et SR.cpp de SR-lib.

  python recomp/leia/construire.py [--sr-lib DOSSIER] [--out FICHIER.dll] [--sans-sr]

SR-lib (github.com/bo3b/SR-lib, licence MIT) apporte SR.cpp et le SDK SR de Leia (SR-SDK-<version>/x64 : en-têtes et
bibliothèques d'import seulement ; les DLL SR restent celles de la SR Platform installée chez le joueur, jamais
redistribuées). Dossier cherché dans l'ordre : --sr-lib, la variable SAR_SR_LIB_ROOT, recomp/leia/SR-lib (ignoré par
git). Sans SR-lib, ou avec --sans-sr, le module est construit sans le SDK : il ne sait alors que le tissage simulé
(SAR_LEIA_SIMULATE=1), qui vérifie le chemin Vulkan → Direct3D 12 → Vulkan sans écran SR.

Toutes les DLL SR (et OpenCV) sont chargées en différé (/DELAYLOAD) : le module se charge, et son create() échoue
proprement, sur une machine sans la SR Platform. Il faut les outils de compilation MSVC x64 (Visual Studio ou Build
Tools, trouvés par vswhere) ; le jeu, lui, reste construit avec GCC / Clang et charge ce module à l'exécution.
"""
import argparse
import os
import re
import subprocess
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
RECOMP = os.path.dirname(ICI)
SOURCE = os.path.join(ICI, "sar_leia_sr.cpp")
ABI = os.path.join(RECOMP, "runtime", "os")
SORTIE = os.path.join(RECOMP, "build", "full", "sar_leia_sr.dll")
OBJ = os.path.join(ICI, "obj")

# Comme SRLIB_DELAYLOAD_DLLS de SR-lib (CMakeLists.txt) : la liste complète, chargée en différé
DLL_SR = ["simulatedreality.dll", "SimulatedRealityCore.dll", "SimulatedRealityCameras.dll",
          "SimulatedRealityHandTrackers.dll", "SimulatedRealityFaceTrackers.dll", "SimulatedRealityUserModelers.dll",
          "SimulatedRealityDisplays.dll", "SimulatedRealityDirectX.dll", "SimulatedRealityOpenGL.dll",
          "DimencoWeaving.dll", "opencv_world343.dll"]
LIB_SR = ["simulatedreality", "SimulatedRealityCore", "SimulatedRealityCameras", "SimulatedRealityHandTrackers",
          "SimulatedRealityFaceTrackers", "SimulatedRealityUserModelers", "SimulatedRealityDisplays",
          "SimulatedRealityDirectX", "SimulatedRealityOpenGL", "DimencoWeaving"]


def echec(msg):
    print("construire leia: " + msg, file=sys.stderr)
    sys.exit(1)


def vcvars():
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    r = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                       capture_output=True, text=True)
    chemin = os.path.join(r.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat") if r.stdout.strip() else ""
    return chemin if chemin and os.path.exists(chemin) else None


def sr_lib(arg):
    for c in [arg, os.environ.get("SAR_SR_LIB_ROOT"), os.path.join(ICI, "SR-lib")]:
        if c and os.path.exists(os.path.join(c, "SR.cpp")) and os.path.exists(os.path.join(c, "SR.hpp")):
            return c
    return None


def sdk(racine):
    versions = sorted((d for d in os.listdir(racine) if re.fullmatch(r"SR-SDK-[0-9.]+", d)),
                      key=lambda d: [int(x) for x in d[7:].split(".") if x.isdigit()])
    if not versions:
        return None
    x64 = os.path.join(racine, versions[-1], "x64")
    return x64 if os.path.exists(os.path.join(x64, "lib", "SimulatedRealityCore.lib")) else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sr-lib")
    p.add_argument("--out", default=SORTIE)
    p.add_argument("--sans-sr", action="store_true")
    a = p.parse_args()
    vc = vcvars()
    if not vc:
        echec("vcvars64.bat not found: install the MSVC x64 build tools (Visual Studio or Build Tools)")
    racine = None if a.sans_sr else sr_lib(a.sr_lib)
    x64 = sdk(racine) if racine else None
    avec_sr = bool(racine and x64)
    if avec_sr:
        print(f"SR-lib: {racine}\nSR SDK: {x64}")
    else:
        print("SR-lib / SR SDK not used: building the simulate-only module (SAR_LEIA_SIMULATE)."
              + ("" if a.sans_sr else " Set SAR_SR_LIB_ROOT or pass --sr-lib for the real weave."))
    os.makedirs(OBJ, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    defs = ["/DUNICODE", "/D_UNICODE", "/DNOMINMAX", "/D_CRT_SECURE_NO_WARNINGS"] + (["/DSAR_AVEC_SR"] if avec_sr else [])
    inc = [f'/I"{ABI}"']
    if avec_sr:
        inc += [f'/I"{racine}"', f'/I"{os.path.join(x64, "include")}"']
        for t in ("asio/include", "GLog/include", "OpenCV/include", "rapidxml", "Leap/include"):
            d = os.path.join(x64, "third_party", *t.split("/"))
            if os.path.exists(d):
                inc.append(f'/I"{d}"')
    cl = " ".join(["cl /nologo /std:c++17 /EHsc /O2 /W3 /MD /DNDEBUG"] + defs + inc)
    objs = [os.path.join(OBJ, "sar_leia_sr.obj")]
    lignes = ["@echo off", f'call "{vc}" >nul 2>&1',
              f'{cl} /c "{SOURCE}" /Fo"{objs[0]}"', "if errorlevel 1 exit /b %errorlevel%"]
    libs = ["d3d12.lib", "dxgi.lib", "user32.lib"]
    if avec_sr:
        objs.append(os.path.join(OBJ, "SR.obj"))
        lignes += [f'{cl} /c "{os.path.join(racine, "SR.cpp")}" /Fo"{objs[1]}"', "if errorlevel 1 exit /b %errorlevel%"]
        for n in LIB_SR:
            f = os.path.join(x64, "lib", n + ".lib")
            if os.path.exists(f):
                libs.append(f'"{f}"')
        for f in (os.path.join(x64, "third_party", "OpenCV", "lib", "x64", "opencv_world343.lib"),
                  os.path.join(x64, "third_party", "GLog", "lib", "x64", "glog.lib"),
                  os.path.join(x64, "third_party", "Leap", "lib", "LeapC.lib")):
            if os.path.exists(f):
                libs.append(f'"{f}"')
        libs += ["opengl32.lib", "delayimp.lib"] + [f"/DELAYLOAD:{d}" for d in DLL_SR] + ["/ignore:4199"]
    lignes += [" ".join(["link /NOLOGO /DLL", f'/OUT:"{os.path.abspath(a.out)}"'] + [f'"{o}"' for o in objs] + libs),
               "if errorlevel 1 exit /b %errorlevel%"]
    script = os.path.join(OBJ, "construire.cmd")
    with open(script, "w", encoding="mbcs" if os.name == "nt" else "utf-8") as f:
        f.write("\r\n".join(lignes) + "\r\n")
    r = subprocess.run(["cmd.exe", "/d", "/c", script])
    if r.returncode or not os.path.exists(a.out):
        echec(f"MSVC build failed (exit {r.returncode})")
    print(f"built: {os.path.abspath(a.out)} ({os.path.getsize(a.out)} bytes, SR SDK {'yes' if avec_sr else 'no: simulate only'})")


if __name__ == "__main__":
    main()
