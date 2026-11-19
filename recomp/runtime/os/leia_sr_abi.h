// SPDX-License-Identifier: MIT
// Copyright (c) 2026 agrofubris. Sous licence MIT, pas GPL : le module se lie au SDK SR (propriétaire) de Leia ; texte
// complet de la licence MIT : LICENCES/SR-lib-LICENSE.txt (même texte, autre titulaire).
// Leia SR (Simulated Reality) : interface C entre le jeu (recomp_main.exe, Vulkan) et sar_leia_sr.dll, le module
// optionnel qui lie le SDK SR de Leia (recomp/leia/sar_leia_sr.cpp, construit à part avec MSVC contre SR-lib).
//
// Le jeu rend en Vulkan ; le SDK ne tisse qu'en Direct3D 9 à 12 ou OpenGL. Le module fait donc tout le Direct3D 12 :
// un périphérique sur la carte que Vulkan utilise (même LUID), sa file, la paire et l'image tissée partagées
// (B8G8R8A8_UNORM, poignées NT que Vulkan importe), une barrière (fence) partagée que Vulkan importe comme sémaphore
// chronologique, et le tissage lui-même, appelé comme dans les portages vérifiés sur écran SR (vf2_leia_sr.dll,
// conker_leia_sr.dll) : conversion sRGB coupée des deux côtés, viewport et scissor posés sur la liste de commandes juste
// avant Weave(), taille et format de l'entrée relus à chaque image. L'exécutable du jeu n'a ainsi ni import SR ni
// en-tête Direct3D : sans le module, ou sans la SR Platform qu'il charge en différé, la sortie Leia SR présente la
// paire côte à côte telle quelle.
//
// Le module exporte une fonction, sar_leia_sr_get_api(version), qui rend cette structure pour la version qu'il
// implémente et nullptr sinon. Les deux côtés doivent en avoir la même disposition : changer la version à chaque
// modification.
#pragma once
#include <cstdint>

// 2 (2026-11-19) : view_size, et resize sépare la taille d'une vue de celle de l'image tissée
#define SAR_LEIA_SR_ABI_VERSION 2
#define SAR_LEIA_SR_GET_API "sar_leia_sr_get_api"

struct sar_leia_sr_api_v2 {
    uint32_t version;   // SAR_LEIA_SR_ABI_VERSION
    uint32_t pad;

    // luid : les 8 octets de VkPhysicalDeviceIDProperties::deviceLUID (la carte de Vulkan) ; window : le HWND présenté
    // (le tisseur lit sa position sur l'écran). Crée le périphérique D3D12, sa file, le tisseur et la barrière partagée
    // (*out_fence : poignée NT, à fermer par l'appelant après import). false : pas de runtime SR, de service ou
    // d'écran SR (ou pas de D3D12 sur cette carte) ; l'appelant n'essaie plus.
    bool (*create)(const uint8_t* luid, void* window, void** out_handle, void** out_fence);

    // Taille d'une vue conseillée par le SDK (SR::Display::getRecommendedViewsTextureWidth/Height, lue comme dans
    // l'exemple directx12_weaving du SDK) : le tisseur attend une paire de 2 × w × h. Par défaut le SDK y met la moitié
    // de la résolution physique de l'écran dans chaque sens (1920 × 1080 sur un écran SR 4K). false : inconnue.
    bool (*view_size)(void* handle, uint32_t* w, uint32_t* h);

    // (Re)crée la paire (2 × view_w × view_h, œil gauche à gauche) et l'image tissée (width × height : la fenêtre),
    // B8G8R8A8_UNORM, partagées : *out_pair et *out_woven, poignées NT (D3D12 resource) à fermer par l'appelant après
    // import. À appeler GPU au repos (aucun travail Vulkan ni D3D12 en vol sur les anciennes).
    bool (*resize)(void* handle, uint32_t view_w, uint32_t view_h, uint32_t width, uint32_t height, void** out_pair,
                   void** out_woven);

    // Côté GPU, sur la file D3D12 : attendre que la barrière atteigne wait_value (paire écrite par Vulkan), tisser la
    // paire dans l'image tissée, puis signaler signal_value. Le signal est posé même quand le tissage échoue, pour que
    // Vulkan n'attende jamais en vain. false : ce tissage a échoué (l'appelant abandonne la sortie Leia).
    bool (*weave)(void* handle, uint64_t wait_value, uint64_t signal_value);

    // Attend la fin du travail du module, puis libère le tisseur, le contexte SR et les objets D3D12.
    void (*destroy)(void* handle);

    // Journal : chaque ligne du module (« [LeiaSR] ... ») passe par log (le journal du jeu) ; nullptr : stderr. À poser
    // avant create.
    void (*set_log)(void (*log)(const char* line));
};

typedef const void* (*sar_leia_sr_get_api_fn)(uint32_t version);
