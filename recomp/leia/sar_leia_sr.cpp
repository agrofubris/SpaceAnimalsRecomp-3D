// SPDX-License-Identifier: MIT
// Copyright (c) 2026 agrofubris. Sous licence MIT, pas GPL : le module se lie au SDK SR (propriétaire) de Leia ; texte
// complet de la licence MIT : LICENCES/SR-lib-LICENSE.txt (même texte, autre titulaire).
// sar_leia_sr.dll : la sortie Leia SR (Simulated Reality) de SpaceAnimalsRecomp, pour les écrans à suivi des yeux
// (Acer SpatialLabs, Samsung Odyssey 3D, Lenovo ThinkPad 3D...). Construit avec MSVC (recomp/leia/construire.py) contre
// le SDK SR de Leia au travers de SR-lib (le wrapper de bo3b, licence MIT), qui sonde le runtime SR avant d'y toucher et
// change les exceptions du SDK en HRESULT. Les DLL SR sont chargées en différé par ce module seulement : l'exécutable du
// jeu n'a aucun import SR, et une machine sans la SR Platform obtient un create() en échec, puis la paire côte à côte.
//
// Interface : recomp/runtime/os/leia_sr_abi.h (version 2). Le jeu rend en Vulkan ; ce module fait tout le Direct3D 12
// (périphérique sur la même carte, file, ressources et barrière partagées, tissage), le jeu importe les poignées NT.
//
// Le tissage suit l'appel vérifié sur écran SR dans les autres portages (vf2_leia_sr.dll, 2026-10-05 ; conker_leia_sr.dll) :
// le tisseur créé avant l'initialisation du contexte (SR-lib fait les deux), taille et format de l'entrée relus sur la
// ressource à chaque image, format de sortie posé seulement quand il change, viewport et scissor posés sur la liste de
// commandes elle-même (le viewport du tisseur ne pilote pas le rastériseur), conversion sRGB coupée (UNORM des deux côtés :
// la paire est déjà encodée en gamma), une image de latence et le « late latching » (réglages par défaut de SR-lib).
//
// Diagnostic : SAR_LEIA_SIMULATE=1 fait réussir create() sans runtime SR et remplace le tissage par une copie connue (la
// moitié gauche de l'œil gauche à côté de la moitié droite de l'œil droit) : tout le chemin Vulkan → D3D12 → Vulkan se
// vérifie au pixel près sans écran SR. Construit sans SR-lib (SAR_AVEC_SR non défini), le module ne sait que simuler.

#include "leia_sr_abi.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#ifdef SAR_AVEC_SR
#include "SR.hpp"
#include "sr/management/srcontext.h"
#include "sr/world/display/display.h"
#endif

#include <cstdarg>

namespace {
using Microsoft::WRL::ComPtr;

void (*g_log)(const char*) = nullptr;   // journal du jeu (set_log), sinon stderr
void journal(const char* fmt, ...) {
    char ligne[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(ligne, sizeof ligne, fmt, ap);
    va_end(ap);
    if (g_log) g_log(ligne);
    else std::fprintf(stderr, "%s\n", ligne);
}

// Un allocateur ne se réinitialise qu'une fois sa liste exécutée : un anneau de listes, chaque emplacement gardé par la
// valeur de barrière de sa dernière soumission.
constexpr int Anneau = 3;
constexpr DXGI_FORMAT Format = DXGI_FORMAT_B8G8R8A8_UNORM;

struct Leia {
#ifdef SAR_AVEC_SR
    SimulatedReality::SRInterfaceDX12* sr = nullptr;
#endif
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocators[Anneau];
    ComPtr<ID3D12GraphicsCommandList> lists[Anneau];
    ComPtr<ID3D12Fence> fence;       // partagée avec Vulkan (sémaphore chronologique)
    HANDLE event = nullptr;
    UINT64 slotValue[Anneau] = {};
    UINT64 lastSignal = 0;
    unsigned ring = 0;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12Resource> pair, woven;
    uint32_t width = 0, height = 0;         // image tissée (la fenêtre)
    uint32_t viewW = 0, viewH = 0;          // une vue de la paire (2 × viewW × viewH)
    uint32_t conseilW = 0, conseilH = 0;    // taille de vue conseillée par le SDK (0 : inconnue)
    bool outputFormatSet = false, loggedFirstWeave = false, simulate = false;

    bool waitFor(UINT64 value) {
        if (value == 0 || fence->GetCompletedValue() >= value) return true;
        ResetEvent(event);
        if (FAILED(fence->SetEventOnCompletion(value, event))) return false;
        return WaitForSingleObject(event, 2000) == WAIT_OBJECT_0;
    }

    ~Leia() {
        // Ce module peut encore avoir des listes en vol : attendues avant de tout rendre. Le SDK peut lever une exception
        // sur un périphérique perdu ; rien ne sort de l'interface C.
        if (fence && event) waitFor(lastSignal);
        if (event) CloseHandle(event);
        pair.Reset(), woven.Reset();
        for (int i = 0; i < Anneau; i++) lists[i].Reset(), allocators[i].Reset();
#ifdef SAR_AVEC_SR
        try {
            if (sr) sr->Delete();   // détruit le tisseur, puis le contexte, dans cet ordre
        } catch (...) {
        }
        sr = nullptr;
#endif
        rtvHeap.Reset(), fence.Reset(), queue.Reset(), device.Reset();
    }
};

void transition(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES avant, D3D12_RESOURCE_STATES apres) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = avant;
    b.Transition.StateAfter = apres;
    l->ResourceBarrier(1, &b);
}

bool creerObjets(Leia& m) {
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(m.device->CreateCommandQueue(&qd, IID_PPV_ARGS(m.queue.GetAddressOf())))) return false;
    for (int i = 0; i < Anneau; i++) {
        if (FAILED(m.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(m.allocators[i].GetAddressOf()))))
            return false;
        if (FAILED(m.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m.allocators[i].Get(), nullptr,
                                               IID_PPV_ARGS(m.lists[i].GetAddressOf()))))
            return false;
        m.lists[i]->Close();   // ouverte à la création ; réinitialisée à son premier tour
    }
    if (FAILED(m.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(m.fence.GetAddressOf())))) return false;
    const D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(m.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(m.rtvHeap.GetAddressOf())))) return false;
    m.event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    return m.event != nullptr;
}

// Texture partagée : tas par défaut, D3D12_HEAP_FLAG_SHARED, état COMMON entre deux tissages (Vulkan l'écrit ou la lit
// entre-temps) ; cible de rendu permise (la sortie du tisseur, et l'usage de transfert côté Vulkan).
bool creerPartagee(Leia& m, uint32_t w, uint32_t h, ComPtr<ID3D12Resource>& r, HANDLE* poignee) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w, d.Height = h, d.DepthOrArraySize = 1, d.MipLevels = 1;
    d.Format = Format;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (FAILED(m.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &d, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(r.ReleaseAndGetAddressOf()))))
        return false;
    return SUCCEEDED(m.device->CreateSharedHandle(r.Get(), nullptr, GENERIC_ALL, nullptr, poignee));
}

// Le tissage simulé (SAR_LEIA_SIMULATE), sans mise à l'échelle : la sortie effacée en noir, puis, depuis son coin haut
// gauche, la moitié gauche de l'œil gauche à côté de la moitié droite de l'œil droit (vues de vw × vh, coupées à la
// sortie w × h si elles la dépassent).
void tissageSimule(Leia& m, ID3D12GraphicsCommandList* l, uint32_t vw, uint32_t vh, uint32_t w, uint32_t h) {
    ID3D12Resource* paire = m.pair.Get();
    ID3D12Resource* sortie = m.woven.Get();
    transition(l, sortie, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = m.rtvHeap->GetCPUDescriptorHandleForHeapStart();
    m.device->CreateRenderTargetView(sortie, nullptr, rtv);
    const float noir[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    l->ClearRenderTargetView(rtv, noir, 0, nullptr);
    transition(l, sortie, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
    transition(l, paire, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
    src.pResource = paire, src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = sortie, dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    const UINT cw = vw < w ? vw : w, ch = vh < h ? vh : h, moitie = cw / 2;
    const D3D12_BOX g = {0, 0, 0, moitie, ch, 1};
    l->CopyTextureRegion(&dst, 0, 0, 0, &src, &g);
    const D3D12_BOX d = {vw + moitie, 0, 0, vw + cw, ch, 1};
    l->CopyTextureRegion(&dst, moitie, 0, 0, &src, &d);
    transition(l, sortie, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    transition(l, paire, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

// SAR_LEIA_SIMULATE_VIEW=LxH (essai) : la taille de vue que le SDK simulé conseille ; absente : inconnue (le jeu prend
// alors son repli, la moitié de la fenêtre).
void conseilSimule(Leia& m) {
    unsigned w = 0, h = 0;
    if (const char* e = std::getenv("SAR_LEIA_SIMULATE_VIEW"); e && std::sscanf(e, "%ux%u", &w, &h) == 2 && w && h)
        m.conseilW = w, m.conseilH = h;
}

#ifdef SAR_AVEC_SR
// Taille de vue conseillée par le SDK, lue comme dans son exemple directx12_weaving (SR::Display sur un contexte). SR-lib
// garde son contexte pour lui : un contexte à part, le temps de la lecture, appelé une fois le tisseur créé (les DLL SR
// sont alors chargées : pas d'exception de chargement différé possible ici).
void lireConseil(Leia& m) {
    SR::SRContext* ctx = nullptr;
    try {
        ctx = SR::SRContext::create();
        if (SR::Display* d = ctx ? SR::Display::create(*ctx) : nullptr) {
            const int w = d->getRecommendedViewsTextureWidth(), h = d->getRecommendedViewsTextureHeight();
            journal("[LeiaSR] display: %dx%d (physical %dx%d), recommended view %dx%d.", d->getResolutionWidth(),
                    d->getResolutionHeight(), d->getPhysicalResolutionWidth(), d->getPhysicalResolutionHeight(), w, h);
            if (w > 0 && h > 0) m.conseilW = uint32_t(w), m.conseilH = uint32_t(h);
        }
    } catch (...) {
        journal("[LeiaSR] the SR runtime threw while reading the recommended view size.");
    }
    if (ctx) {
        try {
            SR::SRContext::deleteSRContext(ctx);
        } catch (...) {
        }
    }
}
#endif
}   // namespace

extern "C" {
static bool __cdecl leiaCreate(const uint8_t* luid, void* window, void** outHandle, void** outFence) {
    if (!outHandle || !outFence) return false;
    *outHandle = nullptr, *outFence = nullptr;
    if (!luid || !window) return false;
    try {
        auto m = std::make_unique<Leia>();
        m->simulate = std::getenv("SAR_LEIA_SIMULATE") != nullptr;
#ifndef SAR_AVEC_SR
        if (!m->simulate) {
            journal("[LeiaSR] module built without the SR SDK: only SAR_LEIA_SIMULATE works.");
            return false;
        }
#endif
        ComPtr<IDXGIFactory4> fabrique;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(fabrique.GetAddressOf())))) return false;
        LUID l;
        std::memcpy(&l, luid, sizeof l);
        ComPtr<IDXGIAdapter1> carte;
        if (FAILED(fabrique->EnumAdapterByLuid(l, IID_PPV_ARGS(carte.GetAddressOf())))) {
            journal("[LeiaSR] the Vulkan graphics card has no DXGI adapter (LUID not found).");
            return false;
        }
        if (std::getenv("SAR_LEIA_DEBUG")) {   // couche de débogage D3D12 (outils graphiques de Windows installés)
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(dbg.GetAddressOf())))) dbg->EnableDebugLayer();
        }
        if (FAILED(D3D12CreateDevice(carte.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(m->device.GetAddressOf())))) {
            journal("[LeiaSR] no Direct3D 12 device on this graphics card.");
            return false;
        }
        if (!creerObjets(*m)) {
            journal("[LeiaSR] couldn't create the weave's Direct3D 12 objects.");
            return false;
        }
        if (m->simulate) {
            journal("[LeiaSR] SAR_LEIA_SIMULATE: no SR runtime used, the weave is a known copy.");
            conseilSimule(*m);
        } else {
#ifdef SAR_AVEC_SR
            // E_NOINTERFACE (ou un autre échec) sans SR Platform, sans son service ou sans écran SR : le cas courant
            const HRESULT hr = SimulatedReality::CreateSRInterfaceDX12(m->device.Get(), static_cast<HWND>(window), &m->sr);
            if (FAILED(hr) || !m->sr) {
                journal("[LeiaSR] no SR runtime, service or display here (CreateSRInterfaceDX12 hr 0x%08lX).",
                             static_cast<unsigned long>(hr));
                m->sr = nullptr;
                return false;
            }
            m->sr->SetShaderSRGBConversion(false, false);
            lireConseil(*m);
#endif
        }
        HANDLE f = nullptr;
        if (FAILED(m->device->CreateSharedHandle(m->fence.Get(), nullptr, GENERIC_ALL, nullptr, &f))) return false;
        journal("[LeiaSR] weaver initialized.");
        *outFence = f;
        *outHandle = m.release();
        return true;
    } catch (...) {
        journal("[LeiaSR] the SR runtime threw while initializing.");
        return false;
    }
}

static bool __cdecl leiaViewSize(void* handle, uint32_t* w, uint32_t* h) {
    const Leia* m = static_cast<const Leia*>(handle);
    if (!m || !w || !h || !m->conseilW || !m->conseilH) return false;
    *w = m->conseilW, *h = m->conseilH;
    return true;
}

static bool __cdecl leiaResize(void* handle, uint32_t viewW, uint32_t viewH, uint32_t width, uint32_t height,
                               void** outPair, void** outWoven) {
    Leia* m = static_cast<Leia*>(handle);
    if (!m || !outPair || !outWoven || !viewW || !viewH || !width || !height) return false;
    *outPair = *outWoven = nullptr;
    try {
        m->waitFor(m->lastSignal);
        HANDLE p = nullptr, t = nullptr;
        if (!creerPartagee(*m, 2 * viewW, viewH, m->pair, &p)) return false;
        if (!creerPartagee(*m, width, height, m->woven, &t)) {
            CloseHandle(p);
            return false;
        }
        m->viewW = viewW, m->viewH = viewH, m->width = width, m->height = height, m->loggedFirstWeave = false;
        *outPair = p, *outWoven = t;
        return true;
    } catch (...) {
        return false;
    }
}

static bool __cdecl leiaWeave(void* handle, uint64_t waitValue, uint64_t signalValue) {
    Leia* m = static_cast<Leia*>(handle);
    if (!m || !m->queue) return false;
    bool ok = false;
    try {
        // GPU : la paire est écrite par Vulkan jusqu'à waitValue
        if (SUCCEEDED(m->queue->Wait(m->fence.Get(), waitValue)) && m->pair && m->woven) {
            const unsigned slot = m->ring++ % Anneau;
            ID3D12CommandAllocator* a = m->allocators[slot].Get();
            ID3D12GraphicsCommandList* l = m->lists[slot].Get();
            if (m->waitFor(m->slotValue[slot]) && SUCCEEDED(a->Reset()) && SUCCEEDED(l->Reset(a, nullptr))) {
                const uint32_t w = m->width, h = m->height;
                if (m->simulate) {
                    tissageSimule(*m, l, m->viewW, m->viewH, w, h);
                } else {
#ifdef SAR_AVEC_SR
                    transition(l, m->pair.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                    transition(l, m->woven.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
                    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = m->rtvHeap->GetCPUDescriptorHandleForHeapStart();
                    m->device->CreateRenderTargetView(m->woven.Get(), nullptr, rtv);
                    const D3D12_VIEWPORT vp = {0.0f, 0.0f, float(w), float(h), 0.0f, 1.0f};
                    const D3D12_RECT sc = {0, 0, LONG(w), LONG(h)};
                    l->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
                    l->RSSetViewports(1, &vp);
                    l->RSSetScissorRects(1, &sc);
                    // reposée à chaque image : SR-lib relit taille et format sur la ressource (une vue gardée
                    // d'avant un redimensionnement serait périmée)
                    m->sr->SetInputTexture(m->pair.Get());
                    if (!m->outputFormatSet) m->sr->SetOutputFormat(Format), m->outputFormatSet = true;
                    m->sr->Weave(l, vp, sc);
                    transition(l, m->woven.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
                    transition(l, m->pair.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
#endif
                }
                if (SUCCEEDED(l->Close())) {
                    ID3D12CommandList* ls[] = {l};
                    m->queue->ExecuteCommandLists(1, ls);
                    m->slotValue[slot] = signalValue;
                    ok = true;
                    if (!m->loggedFirstWeave) {
                        journal("[LeiaSR] weaving %ux%u (pair %ux%u)%s.", w, h, 2 * m->viewW, m->viewH,
                                     m->simulate ? ", simulated" : "");
                        m->loggedFirstWeave = true;
                    }
                }
            }
        }
    } catch (...) {
        journal("[LeiaSR] the SR runtime threw while weaving.");
        ok = false;
    }
    // toujours signalé : Vulkan attend cette valeur avant de lire l'image tissée
    try {
        m->queue->Signal(m->fence.Get(), signalValue);
        m->lastSignal = signalValue;
    } catch (...) {
    }
    if (ok && FAILED(m->device->GetDeviceRemovedReason())) {
        journal("[LeiaSR] Direct3D 12 device removed (0x%08lX).",
                     static_cast<unsigned long>(m->device->GetDeviceRemovedReason()));
        ok = false;
    }
    return ok;
}

static void __cdecl leiaSetLog(void (*log)(const char*)) { g_log = log; }

static void __cdecl leiaDestroy(void* handle) {
    try {
        delete static_cast<Leia*>(handle);
    } catch (...) {
    }
}

// Le seul nom que le jeu demande. La structure est statique : les fonctions vivent dans ce module, que le jeu garde
// chargé tant qu'il s'en sert.
__declspec(dllexport) const void* __cdecl sar_leia_sr_get_api(uint32_t version) {
    static const sar_leia_sr_api_v2 api = {SAR_LEIA_SR_ABI_VERSION, 0, &leiaCreate, &leiaViewSize, &leiaResize, &leiaWeave,
                                             &leiaDestroy, &leiaSetLog};
    return version == SAR_LEIA_SR_ABI_VERSION ? static_cast<const void*>(&api) : nullptr;
}
}
