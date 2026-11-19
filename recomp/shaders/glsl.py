"""Traduction SASS SM 8.x → GLSL Vulkan (compilé ensuite en SPIR-V par glslangValidator).

Principes :
  * registres R0..Rn, UR0..UR63 : variables `uint` ; prédicats P0..P6, UP0..UP6 : `bool` ;
  * flot de contrôle quelconque sans structuration : chaque cible de branchement ouvre un bloc gardé
    `if (pc == N) do { … } while (false);`, les blocs s'enchaînent dans l'ordre des adresses (en_chaine) ; un
    shader sans saut arrière ni saut calculé s'exécute en une passe, sans boucle ni aiguillage ;
  * constantes `c[banque][octet]` : un tampon de stockage par (étage, banque), set 0,
    binding = étage × 32 + banque ; poignées de textures lues dans ces tampons (comme le matériel),
    puis indexation de tableaux de descripteurs (set 1) : id texture = bits 0-19, échantillonneur =
    bits 20-31 ;
  * attributs (adresses Maxwell) : 0x70 position, 0x80 + 16·i générique i, 0x2f8/0x2fc instance /
    sommet ; sorties de fragments dans R0.. à EXIT selon l'OMAP du SPH.

Contrat de liaisons avec le runtime : voir README (Phase 4).
"""
import os
import re
import struct

from sass import Reg, Imm, CBuf, Pred, RZ, URZ, PT, decode

# numérotation des étages de NVN (NVNshaderStage) : liaisons nvnCommandBufferBind*(étage, …)
STAGE_IDX = {"vert": 0, "frag": 1, "geom": 2, "tesc": 3, "tese": 4, "comp": 5}
# jeu de descripteurs des banques de constantes : sommets 0, fragments 1, calcul 0 ; textures : 2
STAGE_SET = {"vert": 0, "frag": 1, "geom": 0, "tesc": 0, "tese": 0, "comp": 0}
TEX_SET = 2


class Unsupported(Exception):
    pass


def flit(v):
    return f"uintBitsToFloat(0x{v & 0xffffffff:08x}u)"


def flit_f(x):
    """Littéral GLSL d'un flottant exact (puissances de 2 courtes : 0.5, 2.0…)."""
    return repr(float(x))


# Carré de 2×2 pixels. Les shaders de fragments du jeu calculent leurs dérivées d'écran (lissage du texte, niveau de
# détail) en échangeant une valeur avec le pixel voisin du même carré (SHFL.BFLY 1 : voisin horizontal, 2 : vertical),
# puis en la soustrayant selon la voie du pixel dans le carré (FSWZADD). Jusqu'au 2026-11-11, ces échanges étaient
# écrits avec les sous-groupes de Vulkan (subgroupShuffleXor, gl_SubgroupInvocationID & 3). C'est juste sur NVIDIA, où
# le sous-groupe d'un shader de fragments suit les carrés, pixels auxiliaires compris. Sur AMD, un pixel auxiliaire
# (hors du triangle, présent seulement pour les dérivées) n'est pas une voie active : au bord de chaque triangle la
# valeur échangée était indéfinie, et le contour et la diagonale de chaque rectangle ressortaient (« maillage » sur
# le texte et les textures). Écriture portable, mesurée identique sur NVIDIA (sondes du 2026-11-11 : le voisin « 1 »
# est en x ± 1, le voisin « 2 » en y ± 1, la voie vaut parité(x) | parité(y) << 1 en coordonnées de la cible) :
#   voisin horizontal = v ± dFdxFine(v), voisin vertical = v ± dFdyFine(v), signe selon la parité du pixel ;
#   voie = (x & 1) | (y & 1) << 1.
# Les dérivées fines sont définies sur tous les GPU, pixels auxiliaires compris.
# RECOMP_SHADER_QUAD_SUBGROUP=1 : ancienne écriture, pour comparer.
QUAD_PORTABLE = not os.environ.get("RECOMP_SHADER_QUAD_SUBGROUP")
# Votes. 251 shaders de fragments font voter les pixels (VOTE.ALL → subgroupAll) avant de sauter un bloc : sur la
# console, si un seul pixel du groupe a besoin du bloc, tous l'exécutent, pixels auxiliaires compris, pour que les
# dérivées (niveau de détail des textures lues dans le bloc) restent justes. Sur AMD un pixel auxiliaire ne vote pas et
# suit son propre chemin : au bord des triangles, les coordonnées de ses voisins n'étaient pas calculées et les arêtes
# ressortaient en pointillés sur les objets 3D. GL_EXT_shader_quad_control (VK_KHR_shader_quad_control) : full_quads
# fait participer les pixels auxiliaires aux opérations de sous-groupe, quad_derivatives définit les dérivées par carré
# même dans du code conditionnel. Le moteur active l'extension quand le pilote l'offre (pas sur NVIDIA, qui se comporte
# déjà ainsi) et retire ces marques du SPIR-V sinon (vk_render.cpp, spirv_sans_quad_control).
# RECOMP_SHADER_NO_QUAD_CONTROL=1 : traduction sans ces marques.
QUAD_CONTROL = not os.environ.get("RECOMP_SHADER_NO_QUAD_CONTROL")
# Stéréoscopie, convergence automatique (2026-11-16) : mesure de distance des objets transparents de la vue principale
# (hologrammes, verres…), qui n'écrivent pas de profondeur et que la mesure par la profondeur du prepass ne voit pas.
# Le runtime pousse pour les dessins mesurés l'adresse GPU d'un tampon et un numéro d'emplacement (constante poussée des
# sommets, 16 octets ; adresse nulle : rien) ; chaque sommet au centre de l'écran (|x| < 0,5 w, |y| < 0,6 w) y range
# sa distance de vue (w, sur l'axe de la caméra) par minimum atomique, et se compte. Ajoutée après le corps de tous les
# shaders de sommets ; un branchement uniforme quand elle est coupée.
MESURE_SOMMETS = [
    "layout(push_constant) uniform PCV_ { uvec2 adr; uint slot; uint pad; } PCV;",
    "layout(buffer_reference, std430, buffer_reference_align = 4) buffer MESV { uint v[]; };",
    "void sar_mesure() {",
    "  if (PCV.adr == uvec2(0u)) return;",
    "  vec4 p = gl_Position;",
    "  if (!(p.w > 0.0 && abs(p.x) < 0.5 * p.w && abs(p.y) < 0.6 * p.w)) return;",
    "  MESV m = MESV(PCV.adr);",
    "  atomicMin(m.v[2u * PCV.slot], floatBitsToUint(p.w));",
    "  atomicAdd(m.v[2u * PCV.slot + 1u], 1u);",
    "}",
]

# Voie du pixel dans son carré : parité de sa position en fragments, calculée une fois en tête de main (QUAD_VOIE,
# flot uniforme). Position divisée par la taille du fragment (dérivée fine de gl_FragCoord : 1 au taux normal, 2 dans
# l'axe réduit au taux d'ombrage 1×2 ou 2×1 de la stéréo, où gl_FragCoord est le centre du fragment de deux pixels et
# sa partie entière toujours impaire). Au taux normal, x / 1 = x : mêmes voies qu'avant, au bit près (2026-11-17).
QUAD_AIDES = [
    "uint QLANE_V;",
    "#define QLANE QLANE_V",
    "uint quad_x(uint r) { float v = uintBitsToFloat(r); float d = dFdxFine(v);"
    " return floatBitsToUint((QLANE_V & 1u) == 0u ? v + d : v - d); }",
    "uint quad_y(uint r) { float v = uintBitsToFloat(r); float d = dFdyFine(v);"
    " return floatBitsToUint((QLANE_V & 2u) == 0u ? v + d : v - d); }",
]
QUAD_VOIE = ("  QLANE_V = (uint(gl_FragCoord.x / dFdxFine(gl_FragCoord.x)) & 1u)"
             " | ((uint(gl_FragCoord.y / dFdyFine(gl_FragCoord.y)) & 1u) << 1);")

# Cœurs tenseurs (emit_tensor) : fragments du warp de 32 fils, comme mma.sync / ldmatrix du PTX. Fil l :
# g = l >> 2, t = l & 3. IMMA.16832 : A (16×32, s8) reg 0 = ligne g, colonnes 4t..4t+3 ; reg 1 = ligne g+8 ;
# regs 2 et 3 = mêmes lignes, colonnes + 16. B (32×8) reg 0 = colonne g, lignes 4t..4t+3 ; reg 1 = lignes + 16.
# HMMA.16816 : mêmes rôles par paires de demi-flottants (colonnes / lignes 2t, 2t+1 ; + 8 pour les regs 2-3 de A
# et le reg 1 de B). D (16×8) : ligne g puis g+8, colonnes 2t, 2t+1. LDSM : la matrice j a ses 8 lignes aux
# adresses données par les fils 8j..8j+7 ; le fil l en reçoit le mot (l & 3) de la ligne l >> 2.
# Fil du warp = gl_SubgroupInvocationID & 31 (sous-groupes de 32 sur NVIDIA, et de 64 = deux warps ailleurs).
TENSOR_AIDES = [
    "uint wlane() { return gl_SubgroupInvocationID & 31u; }",
    "uint wbase() { return gl_SubgroupInvocationID & ~31u; }",
    "int dp4s8(uint a, uint b) { int s = 0; for (int k = 0; k < 32; k += 8)"
    " s += bitfieldExtract(int(a), k, 8) * bitfieldExtract(int(b), k, 8); return s; }",
    "uvec4 imma16832(uvec4 a, uvec2 b, uvec4 c) {",
    "  uint l = wlane(), wb = wbase(), g = l >> 2u, t = l & 3u; ivec4 d = ivec4(c);",
    "  for (uint u = 0u; u < 4u; u++) {",
    "    uint la = wb + 4u * g + u, l0 = wb + 8u * t + u, l1 = l0 + 4u;",
    "    uint a0 = subgroupShuffle(a.x, la), a1 = subgroupShuffle(a.y, la), a2 = subgroupShuffle(a.z, la),"
    " a3 = subgroupShuffle(a.w, la);",
    "    uint b00 = subgroupShuffle(b.x, l0), b01 = subgroupShuffle(b.y, l0), b10 = subgroupShuffle(b.x, l1),"
    " b11 = subgroupShuffle(b.y, l1);",
    "    d += ivec4(dp4s8(a0, b00) + dp4s8(a2, b01), dp4s8(a0, b10) + dp4s8(a2, b11),"
    " dp4s8(a1, b00) + dp4s8(a3, b01), dp4s8(a1, b10) + dp4s8(a3, b11));",
    "  }",
    "  return uvec4(d);",
    "}",
    # arrondi au plus proche pair en entiers : unpackHalf2x16(packHalf2x16(v)) est replié en identité par le
    # compilateur de NVIDIA (arrondi intermédiaire perdu : 1 ulp d'écart, débordements non saturés en infini)
    "uint f16rne(float f) { uint u = floatBitsToUint(f), s = (u >> 16u) & 0x8000u; u &= 0x7fffffffu;",
    "  if (u > 0x7f800000u) return 0x7fffu;   // NaN canonique du HMMA (mesuré : 0x7fff)",
    "  if (u >= 0x47800000u) return s | 0x7c00u;",
    "  if (u < 0x38800000u) { uint e = u >> 23u; if (e < 102u) return s;",
    "    uint m = (u & 0x7fffffu) | 0x800000u, sh = 126u - e, r = m >> sh, rem = m & ((1u << sh) - 1u),"
    " hf = 1u << (sh - 1u);",
    "    if (rem > hf || (rem == hf && (r & 1u) != 0u)) r++; return s | r; }",
    "  uint b = (u - 0x38000000u) >> 13u, rem = u & 0x1fffu;",
    "  if (rem > 0x1000u || (rem == 0x1000u && (b & 1u) != 0u)) b++; return s | b; }",
    # HMMA.16816.F16, mesuré sur la RTX 3080 (.local/tc, cas d'égalité d'arrondi construits) : deux moitiés k = 0-7
    # puis 8-15 ; dans chacune, l'accumulateur et les 8 produits sont alignés sur le plus grand exposant nominal
    # E (ea + eb pour un produit, avant normalisation ; celui de l'accumulateur), chaque terme est tronqué vers zéro
    # à 2^(E-24), la somme est exacte puis arrondie au plus proche pair en demi-flottant. Infini ou NaN parmi les
    # opérandes : calcul flottant ordinaire.
    "void h16dec(uint h, out int m, out int e) { uint x = (h >> 10u) & 31u, mm = h & 0x3ffu;"
    " m = int(x != 0u ? (mm | 0x400u) : mm); e = int(max(x, 1u)) - 15; if ((h & 0x8000u) != 0u) m = -m; }",
    "int htrunc(int m, int sh) { if (m == 0 || sh <= -31) return 0; uint v = uint(abs(m));"
    " uint r = sh >= 0 ? (v << uint(sh)) : (v >> uint(-sh)); return m < 0 ? -int(r) : int(r); }",
    "uint hfromfix(int sum, int E) { if (sum == 0) return 0u; uint s = sum < 0 ? 0x8000u : 0u, v = uint(abs(sum));",
    "  int n = findMSB(v), ex = max(n + E - 24, -14), sh = ex - E + 14; uint m;",
    "  if (sh > 0) { if (sh >= 32) m = 0u; else { m = v >> uint(sh); uint rem = v & ((1u << uint(sh)) - 1u),"
    " hf = 1u << uint(sh - 1); if (rem > hf || (rem == hf && (m & 1u) != 0u)) m++; } }",
    "  else m = v << uint(-sh);",
    "  int h = ((ex + 14) << 10) + int(m); return h >= 0x7c00 ? (s | 0x7c00u) : (s | uint(h)); }",
    # a, b : 4 mots de deux demi-flottants (k = 2u, 2u+1). Boucles non déroulées ([[dont_unroll]]) : un réseau compte
    # jusqu'à 384 HMMA, chacun copié en ligne par le compilateur du pilote ; déroulées, celui de NVIDIA plantait ou
    # épuisait la mémoire sur les plus gros (059b307b6448e8a1 et trois autres).
    "uint hchunk(uint s, uint a[4], uint b[4]) {",
    "  int ms, es, E = -1000; bool sp = (s & 0x7c00u) == 0x7c00u; h16dec(s, ms, es); if (ms != 0) E = es;",
    "  [[dont_unroll]] for (uint k = 0u; k < 8u; k++) { uint x = (a[k >> 1u] >> (16u * (k & 1u))) & 0xffffu,"
    " y = (b[k >> 1u] >> (16u * (k & 1u))) & 0xffffu; sp = sp || (x & 0x7c00u) == 0x7c00u || (y & 0x7c00u) == 0x7c00u;"
    " int mx, ex, my, ey; h16dec(x, mx, ex); h16dec(y, my, ey); if (mx != 0 && my != 0) E = max(E, ex + ey); }",
    "  if (sp) { float f = unpackHalf2x16(s).x; [[dont_unroll]] for (uint k = 0u; k < 8u; k++)"
    " f += unpackHalf2x16((a[k >> 1u] >> (16u * (k & 1u))) & 0xffffu).x * unpackHalf2x16((b[k >> 1u] >> (16u * (k & 1u))) & 0xffffu).x;"
    " return f16rne(f); }",
    "  if (E == -1000) return 0u;",
    "  int sum = htrunc(ms, es - E + 14);",
    "  [[dont_unroll]] for (uint k = 0u; k < 8u; k++) { int mx, ex, my, ey;"
    " h16dec((a[k >> 1u] >> (16u * (k & 1u))) & 0xffffu, mx, ex); h16dec((b[k >> 1u] >> (16u * (k & 1u))) & 0xffffu, my, ey);"
    " sum += htrunc(mx * my, ex + ey - E + 4); }",
    "  return hfromfix(sum, E);",
    "}",
    "uvec2 hmma16816f16(uvec4 a, uvec2 b, uvec2 c) {",
    "  uint l = wlane(), wb = wbase(), g = l >> 2u, t = l & 3u;",
    "  uint d[4] = uint[4](c.x & 0xffffu, c.x >> 16u, c.y & 0xffffu, c.y >> 16u);   // (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)",
    "  [[dont_unroll]] for (uint h = 0u; h < 2u; h++) {",
    "    uint ar = h == 0u ? a.x : a.z, ar8 = h == 0u ? a.y : a.w, br = h == 0u ? b.x : b.y;",
    "    uint A2[2][4], B2[2][4];   // lignes g, g+8 de A ; colonnes 2t, 2t+1 de B (mots du fil 4g+u, 8t+u, 8t+4+u)",
    "    [[dont_unroll]] for (uint u = 0u; u < 4u; u++) {",
    "      A2[0][u] = subgroupShuffle(ar, wb + 4u * g + u); A2[1][u] = subgroupShuffle(ar8, wb + 4u * g + u);",
    "      B2[0][u] = subgroupShuffle(br, wb + 8u * t + u); B2[1][u] = subgroupShuffle(br, wb + 8u * t + 4u + u);",
    "    }",
    "    [[dont_unroll]] for (uint o = 0u; o < 4u; o++) d[o] = hchunk(d[o], A2[o >> 1u], B2[o & 1u]);",
    "  }",
    "  return uvec2(d[0] | (d[1] << 16u), d[2] | (d[3] << 16u));",
    "}",
    "uint ldsm_row(uint la, uint j) { uint l = wlane();"
    " uint ra = subgroupShuffle(la, wbase() + 8u * j + (l >> 2u)) + 4u * (l & 3u); return SMEM[ra >> 2u]; }",
    # .trans : le fil l reçoit M[2(l&3)][l>>2] et M[2(l&3)+1][l>>2] (lignes de M aux adresses des fils 8j+i)
    "uint smem16(uint b) { return (SMEM[b >> 2u] >> (16u * ((b >> 1u) & 1u))) & 0xffffu; }",
    "uint ldsm_rowt(uint la, uint j) { uint l = wlane(), r = l >> 2u, c = 2u * (l & 3u), b = wbase() + 8u * j + c;"
    " return smem16(subgroupShuffle(la, b) + 2u * r) | (smem16(subgroupShuffle(la, b + 1u) + 2u * r) << 16u); }",
    # MOVM.16.MT88 : fragment 8×8 (ligne l>>2, colonnes 2(l&3), +1) de la transposée
    "uint movm(uint a) { uint l = wlane(), r = l >> 2u, c = 2u * (l & 3u), sh = 16u * (r & 1u);"
    " uint w0 = subgroupShuffle(a, wbase() + 4u * c + (r >> 1u)), w1 = subgroupShuffle(a, wbase() + 4u * (c + 1u) + (r >> 1u));"
    " return ((w0 >> sh) & 0xffffu) | (((w1 >> sh) & 0xffffu) << 16u); }",
]
def COOPMAT_AIDES(nwarps, base=None):
    """HMMA.16816.F16 sur les cœurs tenseurs du GPU (VK_KHR_cooperative_matrix, forme M16 N8 K16 en demi-flottants,
    la même instruction HMMA sur NVIDIA) au lieu de l'émulation : chaque fil range ses fragments dans un brouillon
    partagé par warp (A 16×16 par lignes, B 16×8 par colonnes, C 16×8 par lignes, 256 mots), coopMatLoad,
    coopMatMulAdd, coopMatStore, puis relit son fragment de D. Choisi à l'exécution quand le GPU l'offre
    (vk_render.cpp) ; sinon la variante émulée (TENSOR_AIDES)."""
    # base : premier mot du brouillon dans SMEM (après la mémoire partagée du programme), sinon tableau HMS à part
    decl = [f"shared uint HMS[{256 * max(1, nwarps)}];"] if base is None else [f"#define HMS SMEM"]
    return decl + [
        "uvec2 hmma16816f16(uvec4 a, uvec2 b, uvec2 c) {",
        f"  uint l = gl_SubgroupInvocationID & 31u, g = l >> 2u, t = l & 3u, base = {base or 0}u + gl_SubgroupID * 256u;",
        "  subgroupBarrier();",
        "  HMS[base + g * 8u + t] = a.x; HMS[base + (g + 8u) * 8u + t] = a.y;",
        "  HMS[base + g * 8u + 4u + t] = a.z; HMS[base + (g + 8u) * 8u + 4u + t] = a.w;",
        "  HMS[base + 128u + g * 8u + t] = b.x; HMS[base + 128u + g * 8u + 4u + t] = b.y;",
        "  HMS[base + 192u + g * 4u + t] = c.x; HMS[base + 192u + (g + 8u) * 4u + t] = c.y;",
        "  subgroupBarrier();",
        "  coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mA;",
        "  coopmat<float16_t, gl_ScopeSubgroup, 16, 8, gl_MatrixUseB> mB;",
        "  coopmat<float16_t, gl_ScopeSubgroup, 16, 8, gl_MatrixUseAccumulator> mC;",
        "  coopMatLoad(mA, HMS, base, 8u, gl_CooperativeMatrixLayoutRowMajor);",
        "  coopMatLoad(mB, HMS, base + 128u, 8u, gl_CooperativeMatrixLayoutColumnMajor);",
        "  coopMatLoad(mC, HMS, base + 192u, 4u, gl_CooperativeMatrixLayoutRowMajor);",
        "  mC = coopMatMulAdd(mA, mB, mC);",
        "  coopMatStore(mC, HMS, base + 192u, 4u, gl_CooperativeMatrixLayoutRowMajor);",
        "  subgroupBarrier();",
        "  return uvec2(HMS[base + 192u + g * 4u + t], HMS[base + 192u + (g + 8u) * 4u + t]);",
        "}",
    ]


HMMA_EN_LIGNE = int(os.environ.get("RECOMP_HMMA_EN_LIGNE", "4"))             # HMMA copiés en ligne au plus ; au-delà, un seul bloc partagé (translate)
HMMA_PC = 0xfffffff0         # étiquette de ce bloc (hors des adresses d'instruction, multiples de 16)
HRELU_AIDE = ["uint hrelu1(uint h) { return ((h & 0x8000u) != 0u || (h & 0x7fffu) > 0x7c00u) ? 0u : h; }",
              "uint hrelu(uint x) { return hrelu1(x & 0xffffu) | (hrelu1(x >> 16u) << 16u); }"]


class Emitter:
    def __init__(self, stage):
        self.st = stage
        self.kind = stage.kind
        self.sidx = STAGE_IDX[self.kind]
        raw = stage.insns_bytes
        self.insns = []
        for pc in range(0, len(raw) - 15, 16):
            lo, hi = struct.unpack_from("<QQ", raw, pc)
            self.insns.append(decode(pc, lo | (hi << 64)))
        self.banks = set()
        self.max_r = 0
        self.vin, self.vout, self.fin = set(), set(), {}
        self.uses = set()
        self.img_handles = set()   # (banque, déplacement, dim) des poignées d'images
        self.tex_handles = set()   # (banque, déplacement) des poignées de textures lues en banque constante
        self.warn = []
        self.lines = []

    # ---------------------------------------------------------------- opérandes
    def r(self, n):
        if n == RZ:
            return "0u"
        self.max_r = max(self.max_r, n)
        return f"R{n}"

    def ur(self, n):
        return "0u" if n == URZ else f"UR{n}"

    def cb(self, bank, off_expr):
        if bank == 1 and getattr(self.st, "cb1", b""):
            # banque 1 = constantes du programme (vidées avec le shader) : tableau constant intégré
            self.uses.add("cb1")
            n = len(self.st.cb1) // 4
            return f"CB1C[min(uint({off_expr}), {n - 1}u)]"
        self.banks.add(bank)
        return f"cb{self.sidx}_{bank}.d[{off_expr}]"

    def cbw(self, c: CBuf, extra=0):
        if c.ubank is not None:
            raise Unsupported("constant with indirect bank (UR)")
        return self.cb(c.bank, f"{(c.off + extra) // 4}u")

    def u(self, o):
        """Valeur entière 32 bits d'un opérande (sans modificateur flottant)."""
        if isinstance(o, Reg):
            v = self.ur(o.n) if o.uni else self.r(o.n)
        elif isinstance(o, Imm):
            v = f"0x{o.v:08x}u"
        elif isinstance(o, CBuf):
            v = self.cbw(o)
        else:
            raise Unsupported(f"operand {o!r}")
        return v

    def ui(self, o):
        """Entier avec négation (complément à deux)."""
        v = self.u(o)
        if getattr(o, "neg", False):
            v = f"(0u - {v})"
        return v

    def f(self, o):
        if isinstance(o, Imm):
            v = flit(o.v)
        elif isinstance(o, Reg) and o.zero:
            v = "0.0"
        else:
            v = f"uintBitsToFloat({self.u(o)})"
        if getattr(o, "abs", False):
            v = f"abs({v})"
        if getattr(o, "neg", False):
            v = f"(-{v})"
        return v

    def p(self, pr: Pred):
        if pr.n == PT:
            v = "true"
        else:
            v = f"UP{pr.n}" if pr.uni else f"P{pr.n}"
        return f"(!{v})" if pr.neg else v

    def setp(self, pr: Pred, expr):
        if pr.n == PT:
            return ""
        return f"{'UP' if pr.uni else 'P'}{pr.n} = {expr};"

    def setr(self, d: Reg, expr):
        if d.uni:
            return "" if d.n == URZ else f"UR{d.n} = {expr};"
        if d.n == RZ:
            return ""
        self.max_r = max(self.max_r, d.n)
        return f"R{d.n} = {expr};"

    def setf(self, d, expr, sat=False):
        if sat:
            expr = f"clamp({expr}, 0.0, 1.0)"
        return self.setr(d, f"floatBitsToUint({expr})")

    def rn(self, d: Reg, k):
        return Reg(d.n + k if d.n != RZ else RZ, uni=d.uni)

    # ---------------------------------------------------------------- instructions
    def emit(self, i):
        m = i.mods
        n = i.name
        s = i.srcs
        # BSSY / BSYNC / BREAK ne gèrent que la reconvergence du warp (barrières de convergence) : le
        # flot de contrôle lui-même passe par BRA, donc sans effet pour une traduction par thread.
        if n in ("nop", "bssy", "bsync", "break") or n == "membar" and self.kind != "comp":
            return ""
        if n == "fadd":
            return self.setf(i.dst, f"{self.f(s[0])} + {self.f(s[1])}", m["sat"])
        if n == "fmul":
            # échelle du résultat (champ 84-87) : 4 = ×1 ; 2^(v−4) supposé (3 = .D2, 5 = .M2…)
            pd = m.get("pdiv", 4)
            if pd != 4:
                sc = 2.0 ** (pd - 4)
                return self.setf(i.dst, f"({self.f(s[0])} * {self.f(s[1])}) * {flit_f(sc)}", m["sat"])
            return self.setf(i.dst, f"{self.f(s[0])} * {self.f(s[1])}", m["sat"])
        if n == "ffma":
            return self.setf(i.dst, f"fma({self.f(s[0])}, {self.f(s[1])}, {self.f(s[2])})", m["sat"])
        if n == "fmnmx":
            a, b = self.f(s[0]), self.f(s[1])
            return self.setf(i.dst, f"({self.p(i.psrc[0])} ? min({a}, {b}) : max({a}, {b}))")
        if n in ("fsetp", "fset", "isetp", "hsetp2"):
            return self.emit_setp(i)
        if n == "fsel":
            return self.setr(i.dst, f"({self.p(i.psrc[0])} ? {self.u(s[0])} : {self.u(s[1])})") if not (s[0].neg or s[0].abs or s[1].neg or s[1].abs) \
                else self.setf(i.dst, f"({self.p(i.psrc[0])} ? {self.f(s[0])} : {self.f(s[1])})")
        if n == "mufu":
            # Demi-précision (f16) : la source est la moitié du mot choisie par sub, le résultat un f16 dans la
            # moitié basse. Jusqu'au 2026-11-13 le drapeau était ignoré : le mot entier était lu comme un f32 (un seul
            # shader traduit concerné, d38474b0c297f4ed : quatre inverses d'un flou radial ; les autres cas sont dans
            # des shaders de calcul pas encore traduits).
            f16 = bool(m.get("f16"))
            if f16:
                x = f"unpackHalf2x16({self.u(s[0])}).{'y' if m.get('sub', 0) & 1 else 'x'}"
                if getattr(s[0], "abs", False):
                    x = f"abs({x})"
                if getattr(s[0], "neg", False):
                    x = f"(-{x})"
            else:
                x = self.f(s[0])
            op = m["op"]
            # MUFU.SIN/COS (Volta et après, sans RRO) prennent l'angle en TOURS : le compilateur multiplie
            # d'abord par 1/(2π) (FMUL.RZ 0.15915494). Le matériel prend la partie fractionnaire.
            e = {"cos": f"cos(fract({x}) * 6.28318530718)", "sin": f"sin(fract({x}) * 6.28318530718)",
                 "ex2": f"exp2({x})", "lg2": f"log2({x})",
                 "rcp": f"(1.0 / {x})", "rsq": f"inversesqrt({x})", "sqrt": f"sqrt({x})",
                 "tanh": f"tanh({x})"}.get(op)
            if e is None:
                raise Unsupported(f"mufu {op}")
            if f16:
                return self.setr(i.dst, f"(packHalf2x16(vec2({e}, 0.0)) & 0xffffu)")
            return self.setf(i.dst, e)
        if n == "frnd":
            x = self.f(s[0])
            fn = ["roundEven", "floor", "ceil", "trunc"][m["rnd"]]
            return self.setf(i.dst, f"{fn}({x})")
        if n == "f2i":
            # source : 1 f16 (moitié choisie par sub), 2 f32 ; destination : 0 8 bits, 1 16 bits, 2 32 bits
            # (résultat saturé dans la plage du type, étendu à 32 bits selon le signe)
            if m["st"] not in (1, 2) or m["dt"] not in (0, 1, 2):
                raise Unsupported(f"f2i sizes {m['st']}/{m['dt']}")
            if m["st"] == 1:
                x = f"unpackHalf2x16({self.u(s[0])}).{'y' if m.get('sub', 0) & 1 else 'x'}"
                if getattr(s[0], "abs", False):
                    x = f"abs({x})"
                if getattr(s[0], "neg", False):
                    x = f"(-{x})"
            else:
                x = self.f(s[0])
            fn = ["roundEven", "floor", "ceil", "trunc"][m["rnd"]]
            smin, smax, umax = {0: ("-128.0", "127.0", "255.0"), 1: ("-32768.0", "32767.0", "65535.0"),
                                2: ("-2147483648.0", "2147483520.0", "4294967040.0")}[m["dt"]]
            if m["signed"]:
                return self.setr(i.dst, f"uint(int(clamp({fn}({x}), {smin}, {smax})))")
            return self.setr(i.dst, f"uint(clamp({fn}({x}), 0.0, {umax}))")
        if n in ("i2f", "i2fp"):
            st, dt = m.get("st", 2), m.get("dt", 2)
            v = self.u(s[0])
            if n == "i2f" and st != 2:
                # 8 / 16 bits : l'octet (0 à 3) ou la moitié (0 ou 1) du mot source est choisi par sub, comme pour
                # f2i et f2f. Jusqu'au 2026-11-13, toujours l'octet 0 : les décalcomanies projetées au sol lisaient
                # leur couche d'atlas dans l'octet des drapeaux au lieu de l'octet 2 (taches bleues de Corneria :
                # une carte de normales affichée comme couleur). 252 instructions du jeu ont un sub non nul.
                bitsz = {0: 8, 1: 16}.get(st)
                if bitsz is None:
                    raise Unsupported("i2f 64 bits")
                dec = (m.get("sub", 0) & (3 if st == 0 else 1)) * bitsz
                v = f"bitfieldExtract({'int' if m['signed'] else 'uint'}({v}), {dec}, {bitsz})"
                conv = f"float({v})"
            else:
                conv = f"float(int({v}))" if m["signed"] else f"float({v})"
            if n == "i2f" and dt == 1:
                return self.setr(i.dst, f"(packHalf2x16(vec2({conv}, 0.0)) & 0xffffu)")
            return self.setf(i.dst, conv)
        if n == "f2f":
            st, dt = m["st"], m["dt"]
            if st == 2 and dt == 1:      # f32 → f16 (dans la moitié basse)
                return self.setr(i.dst, f"(packHalf2x16(vec2({self.f(s[0])}, 0.0)) & 0xffffu)")
            if st == 1 and dt == 2:      # f16 → f32 (moitié choisie par sub)
                hi = m.get("sub", 0) & 1
                v = f"unpackHalf2x16({self.u(s[0])}).{'y' if hi else 'x'}"
                if s[0].abs:
                    v = f"abs({v})"
                if s[0].neg:
                    v = f"(-{v})"
                return self.setf(i.dst, v)
            if st == 2 and dt == 2:
                return self.setf(i.dst, self.f(s[0]))
            raise Unsupported(f"f2f {st}->{dt}")
        if n == "f2fp":
            # PACK_AB : A dans la moitié haute, B dans la basse
            return self.setr(i.dst, f"packHalf2x16(vec2({self.f(s[1])}, {self.f(s[0])}))")
        if n in ("hadd2", "hmul2", "hfma2"):
            return self.emit_h2(i)
        if n == "iadd3":
            return self.emit_iadd3(i)
        if n == "imad":
            a, b, c = self.u(s[0]), self.u(s[1]), self.ui(s[2])
            if getattr(s[0], "neg", False) or getattr(s[1], "neg", False):
                a, b = self.ui(s[0]), self.ui(s[1])
            e = f"({a} * {b} + {c})"
            if m.get("x"):
                e = f"({e} + ({self.p(i.psrc[0])} ? 1u : 0u))"
            return self.setr(i.dst, e)
        if n == "imad.wide":
            self.uses.add("int64")
            a, b = self.u(s[0]), self.u(s[1])
            c = s[2]
            if isinstance(c, Reg) and c.uni:   # UIMAD.WIDE (0x2a5 / 0x8a5) : C = paire de registres uniformes
                c64 = "0ul" if c.zero else f"(uint64_t({self.ur(c.n + 1)}) << 32 | uint64_t({self.ur(c.n)}))"
            elif isinstance(c, Reg):
                c64 = "0ul" if c.n == RZ else f"(uint64_t({self.r(c.n + 1)}) << 32 | uint64_t({self.r(c.n)}))"
            elif isinstance(c, CBuf):   # constante 64 bits : deux mots (vérifié sur noyaux CUDA, .local/tc/difftest.py)
                c64 = f"(uint64_t({self.cbw(c, 4)}) << 32 | uint64_t({self.cbw(c)}))"
            else:
                c64 = f"uint64_t({self.u(c)})"
            prod = f"uint64_t(int64_t(int({a})) * int64_t(int({b})))" if m["signed"] else f"(uint64_t({a}) * uint64_t({b}))"
            return f"{{ uint64_t t = {prod} + {c64}; {self.setr(i.dst, 'uint(t)')} {self.setr(self.rn(i.dst, 1), 'uint(t >> 32)')} }}"
        if n == "imad.hi":
            self.uses.add("int64")
            a, b, c = self.u(s[0]), self.u(s[1]), self.u(s[2])
            prod = f"uint64_t(int64_t(int({a})) * int64_t(int({b})))" if m["signed"] else f"(uint64_t({a}) * uint64_t({b}))"
            return self.setr(i.dst, f"(uint({prod} >> 32) + {c})")
        if n == "lop3":
            e = self.lut3(m["lut"], self.u(s[0]), self.u(s[1]), self.u(s[2]))
            out = self.setr(i.dst, e)
            if i.pdst and i.pdst[0].n != PT:
                out = f"{{ uint t = {e}; {self.setr(i.dst, 't')} {self.setp(i.pdst[0], 't != 0u')} }}"
            return out
        if n == "shf":
            return self.emit_shf(i)
        if n == "lea":
            a, b = self.u(s[0]), self.u(s[1])
            if m["neg_a"]:
                a = f"(0u - {a})"
            sh = m["shift"]
            if m["hi"]:
                self.uses.add("int64")
                c = self.u(s[2])
                v = f"uint(((uint64_t({c}) << 32 | uint64_t({a})) << {sh}) >> 32)"
                e = f"({v} + {b})"
                if m["x"]:
                    e = f"({e} + ({self.p(i.psrc[0])} ? 1u : 0u))"
                return self.setr(i.dst, e)
            e = f"(({a} << {sh}) + {b})"
            if i.pdst and i.pdst[0].n != PT:
                return f"{{ uint c; uint t = uaddCarry({a} << {sh}, {b}, c); {self.setr(i.dst, 't')} {self.setp(i.pdst[0], 'c != 0u')} }}"
            return self.setr(i.dst, e)
        if n == "prmt":
            if m["mode"] != 0:
                raise Unsupported(f"prmt mode {m['mode']}")
            a, sel, b = self.u(s[0]), self.u(s[1]), self.u(s[2])
            self.uses.add("prmt")
            return self.setr(i.dst, f"prmt({a}, {sel}, {b})")
        if n == "sel":
            return self.setr(i.dst, f"({self.p(i.psrc[0])} ? {self.u(s[0])} : {self.u(s[1])})")
        if n == "imnmx":
            if m["signed"]:   # (le bit 73 code le signe, pas une valeur absolue)
                a, b = f"int({self.u(s[0])})", f"int({self.u(s[1])})"
            else:
                a, b = self.u(s[0]), self.u(s[1])
            return self.setr(i.dst, f"uint({self.p(i.psrc[0])} ? min({a}, {b}) : max({a}, {b}))")
        if n == "iabs":
            return self.setr(i.dst, f"uint(abs(int({self.u(s[0])})))")
        if n == "sgxt":
            a, b = self.u(s[0]), self.u(s[1])
            if m["signed"]:
                return self.setr(i.dst, f"uint(bitfieldExtract(int({a}), 0, int(min({b}, 32u))))")
            return self.setr(i.dst, f"bitfieldExtract({a}, 0, int(min({b}, 32u)))")
        if n == "bmsk":
            a, b = self.u(s[0]), self.u(s[1])
            return self.setr(i.dst, f"(((({b}) >= 32u) ? 0xffffffffu : ((1u << ({b})) - 1u)) << ({a}))")
        if n == "flo":
            v = self.u(s[0])
            if m.get("not"):
                v = f"(~{v})"
            e = f"uint(findMSB(int({v})))" if m["signed"] else f"uint(findMSB({v}))"
            if m["sh"]:
                e = f"(({v}) == 0u ? 0xffffffffu : 31u - {e})"
            return self.setr(i.dst, e)
        if n == "popc":
            v = self.u(s[0])
            if m.get("not"):
                v = f"(~{v})"
            return self.setr(i.dst, f"uint(bitCount({v}))")
        if n == "brev":
            return self.setr(i.dst, f"bitfieldReverse({self.u(s[0])})")
        if n == "mov":
            return self.setr(i.dst, self.u(s[0]))
        if n in ("umov", "r2ur"):
            return self.setr(i.dst, self.u(s[0]))
        if n == "plop3":
            a, b, c = (self.p(x) for x in i.psrc)
            out = []
            for k, pd in enumerate(i.pdst):
                if pd.n == PT:
                    continue
                lut = m["lut0"] if k == 0 else m["lut1"]
                out.append(self.setp(pd, self.lut3b(lut, a, b, c)))
            return " ".join(out)
        if n == "p2r":
            byte = m.get("byte", 0)
            packed = " | ".join(f"(P{k} ? {1 << k}u : 0u)" for k in range(7))
            return self.setr(i.dst, f"(({self.u(s[0])} & ~(({self.u(s[1])}) << {8 * byte}u)) | ((({packed}) & {self.u(s[1])}) << {8 * byte}u))")
        if n == "r2p":
            byte = m.get("byte", 0)
            out = []
            for k in range(7):
                out.append(f"if ((({self.u(s[1])}) & {1 << k}u) != 0u) P{k} = ((({self.u(s[0])}) >> {8 * byte + k}u) & 1u) != 0u;")
            return " ".join(out)
        if n in ("s2r", "s2ur", "cs2r"):
            return self.emit_sr(i)
        if n == "pixld":   # informations du pixel ; nos cibles ont un seul échantillon
            if self.kind != "frag":
                raise Unsupported("pixld outside fragment")
            v = m["val"]
            if v == 0:      # MS_COUNT : nombre d'échantillons
                return self.setr(i.dst, "1u")
            if v == 1:      # COVMASK : masque de couverture du pixel
                return self.setr(i.dst, "uint(gl_SampleMaskIn[0])")
            raise Unsupported(f"pixld {v}")
        if n == "uldc":
            c = s[0]
            if m["type"] == 5:
                return f"{self.setr(i.dst, self.cbw(c))} {self.setr(self.rn(i.dst, 1), self.cbw(c, 4))}"
            return self.setr(i.dst, self.cbw(c))
        if n == "ldc":
            return self.emit_ldc(i)
        if n in ("ald", "ast", "ipa"):
            return self.emit_attr(i)
        if n in ("tex", "tld", "tld4", "txq"):
            return self.emit_tex(i)
        if n == "txd":
            return self.emit_txd(i)
        if n in ("suld", "sust", "suatom", "sured"):
            return self.emit_surf(i)
        if n in ("lds", "sts", "ldl", "stl", "ldg", "stg", "atoms", "atomg", "red"):
            return self.emit_mem(i)
        if n == "kill":
            if self.kind != "frag":
                raise Unsupported("kill outside fragment")
            return "discard;" if i.psrc[0].true else f"if ({self.p(i.psrc[0])}) discard;"
        if n == "bar":
            return "barrier();"
        if n == "membar":
            self.uses.add("membar")
            return "memoryBarrier();"
        if n == "shfl":
            a = self.u(s[0])
            lane = self.u(s[1])
            if QUAD_PORTABLE and self.kind == "frag" and m["op"] == "bfly" and lane in ("0x00000001u", "0x00000002u"):
                self.uses.add("quadp")   # voisin du carré 2×2, par les dérivées fines (voir QUAD_AIDES)
                out = self.setr(i.dst, f"{'quad_x' if lane == '0x00000001u' else 'quad_y'}({a})")
            else:
                self.uses.add("subgroup")
                fn = {"idx": "subgroupShuffle", "up": "subgroupShuffleUp", "down": "subgroupShuffleDown",
                      "bfly": "subgroupShuffleXor"}[m["op"]]
                out = self.setr(i.dst, f"{fn}({a}, {lane})")
            if i.pdst and i.pdst[0].n != PT:
                out += " " + self.setp(i.pdst[0], "true")
            return out
        if n in ("imma", "hmma", "ldsm", "movm"):
            return self.emit_tensor(i)
        if n == "hrelu":   # HMNMX2 Rd, Ra, RZ (!PT) : max.f16x2(a, 0), voir sass.py
            self.uses.add("hrelu")
            return self.setr(i.dst, f"hrelu({self.u(s[0])})")
        if n == "vote":
            self.uses.add("subgroup")
            pv = self.p(i.psrc[0])
            out = self.setr(i.dst, f"subgroupBallot({pv}).x")
            if i.pdst and i.pdst[0].n != PT:
                fn = {"all": "subgroupAll", "any": "subgroupAny", "eq": "subgroupAllEqual"}[m["op"]]
                out += " " + self.setp(i.pdst[0], f"{fn}({pv})")
            return out
        if n == "redux":   # réduction sur les fils actifs du warp → registre uniforme
            self.uses.update(("subgroup", "subgroup_arith"))
            v = self.u(s[0])
            op, sg = m["op"], m["signed"]
            fn = {0: "subgroupAnd", 1: "subgroupOr", 2: "subgroupXor", 3: "subgroupAdd", 4: "subgroupMin",
                  5: "subgroupMax"}.get(op)
            if fn is None:
                raise Unsupported(f"redux op {op}")
            e = f"uint({fn}(int({v})))" if sg and op in (4, 5) else f"{fn}({v})"
            return self.setr(i.dst, e)
        if n == "atoms.popcinc":   # + nombre de fils actifs (chaque fil ajoute 1) ; valeur rendue non gérée
            if not i.dst.zero:
                raise Unsupported("atoms.popc.inc with a result")
            self.uses.add("shared")
            base, off = s[0], s[1].v
            addr = f"(({self.u(base)} + {off & 0xffffffff}u) >> 2)" if not base.zero else f"{(off & 0xffffff) >> 2}u"
            return f"atomicAdd(SMEM[{addr}], 1u);"
        if n == "fswzadd":
            portable = QUAD_PORTABLE and self.kind == "frag"
            self.uses.add("quadp" if portable else "quad")
            a, b = self.f(s[0]), self.f(s[1])
            sub = m["sub"]
            ops = [(sub >> (2 * (3 - k))) & 3 for k in range(4)]   # lane 0..3
            terms = []
            for k, o in enumerate(ops):
                e = {0: f"({a} + {b})", 1: f"({b} - {a})", 2: f"({a} - {b})", 3: f"{b}"}[o]
                terms.append(e)
            sel = "QLANE" if portable else "(gl_SubgroupInvocationID & 3u)"
            expr = f"({sel} == 0u ? {terms[0]} : {sel} == 1u ? {terms[1]} : {sel} == 2u ? {terms[2]} : {terms[3]})"
            return self.setf(i.dst, expr)
        raise Unsupported(f"instruction {n} (0x{i.opc:03x})")

    # ---------------------------------------------------------------- cœurs tenseurs (IMMA / HMMA / LDSM)
    # Opérations du warp entier, émulées fil par fil avec les mélanges de sous-groupe (TENSOR_AIDES). Sémantique
    # par registre et par fil vérifiée sur une RTX 3080 (2026-11-15) : le PTX équivalent (mma.sync, ldmatrix,
    # movmatrix, max.f16x2) compilé et exécuté par le pilote, comparé bit à bit au GLSL de TENSOR_AIDES sur des
    # données aléatoires et construites (HMMA : alignement, troncature et arrondi décrits avant hchunk).
    def emit_tensor(self, i):
        if self.kind != "comp":
            raise Unsupported(f"{i.name} outside compute")
        self.uses.update(("subgroup", "tensor"))
        d = i.dst
        if i.name == "ldsm":
            self.uses.add("shared")
            n = i.mods["n"]
            a, off = i.srcs[0], i.srcs[1].v
            addr = self.u(a) if not a.zero else "0u"
            if len(i.srcs) > 2 and not i.srcs[2].zero:
                addr = f"({addr} + {self.ur(i.srcs[2].n)})"
            if off:
                addr = f"({addr} + {off}u)"
            fn = "ldsm_rowt" if i.mods.get("trans") else "ldsm_row"
            out = ["{ uint la = " + addr + ";"]
            out += [f"uint t{k} = {fn}(la, {k}u);" for k in range(n)]
            out += [self.setr(self.rn(d, k), f"t{k}") for k in range(n)]
            return " ".join(out) + " }"
        if i.name == "movm":
            return self.setr(d, f"movm({self.u(i.srcs[0])})")
        ra, rb, rc = i.srcs
        regs = lambda r, k: ", ".join(self.u(self.rn(r, j)) for j in range(k))
        if i.name == "hmma":
            self.uses.add("hmma")
        if i.name == "imma":
            call = f"imma16832(uvec4({regs(ra, 4)}), uvec2({regs(rb, 2)}), uvec4({regs(rc, 4)}))"
            nd = 4
        else:
            call = f"hmma16816f16(uvec4({regs(ra, 4)}), uvec2({regs(rb, 2)}), uvec2({regs(rc, 2)}))"
            nd = 2
        comp = "xyzw"
        out = [f"{{ uvec{nd} td = {call};"]
        out += [self.setr(self.rn(d, k), f"td.{comp[k]}") for k in range(nd)]
        return " ".join(out) + " }"

    def lut3(self, lut, a, b, c):
        known = {0x00: "0u", 0xff: "0xffffffffu", 0xf0: a, 0xcc: b, 0xaa: c, 0x0f: f"(~{a})", 0x33: f"(~{b})",
                 0x55: f"(~{c})", 0xc0: f"({a} & {b})", 0xa0: f"({a} & {c})", 0x88: f"({b} & {c})",
                 0xfc: f"({a} | {b})", 0xfa: f"({a} | {c})", 0xee: f"({b} | {c})", 0x3c: f"({a} ^ {b})",
                 0x5a: f"({a} ^ {c})", 0x66: f"({b} ^ {c})", 0x96: f"({a} ^ {b} ^ {c})", 0x80: f"({a} & {b} & {c})",
                 0xfe: f"({a} | {b} | {c})", 0xf8: f"({a} | ({b} & {c}))", 0xe0: f"({a} & ({b} | {c}))",
                 0x30: f"({a} & ~{b})", 0x0c: f"(~{a} & {b})", 0x50: f"({a} & ~{c})", 0x44: f"({b} & ~{c})",
                 0x22: f"(~{b} & {c})", 0x0a: f"(~{a} & {c})", 0xf3: f"({a} | ~{b})", 0xcf: f"(~{a} | {b})",
                 0x3f: f"(~({a} & {b}))", 0xe2: f"(({a} & {b}) | (~{b} & {c}))", 0xb8: f"(({a} & {b}) | ({c} & ~{b}))",
                 0xca: f"(({a} & {b}) | (~{a} & {c}))", 0xd8: f"(({a} & {c}) | (~{c} & {b}))", 0xc8: f"(({a} | {c}) & {b})",
                 0x28: f"(({a} ^ {b}) & {c})", 0x48: f"(({a} ^ {c}) & {b})", 0x60: f"(({b} ^ {c}) & {a})",
                 0x6a: f"(({a} & {b}) ^ {c})", 0x78: f"(({b} & {c}) ^ {a})", 0x1e: f"(({b} | {c}) ^ {a})",
                 0xf6: f"({a} | ({b} ^ {c}))", 0xbe: f"({c} | ({a} ^ {b}))", 0xde: f"({b} | ({a} ^ {c}))"}
        if lut in known:
            return known[lut]
        terms = []
        for k in range(8):
            if lut >> k & 1:
                ta = a if k & 4 else f"~{a}"
                tb = b if k & 2 else f"~{b}"
                tc = c if k & 1 else f"~{c}"
                terms.append(f"({ta} & {tb} & {tc})")
        return "(" + " | ".join(terms) + ")"

    def lut3b(self, lut, a, b, c):
        terms = []
        for k in range(8):
            if lut >> k & 1:
                ta = a if k & 4 else f"!{a}"
                tb = b if k & 2 else f"!{b}"
                tc = c if k & 1 else f"!{c}"
                terms.append(f"({ta} && {tb} && {tc})")
        return "(" + (" || ".join(terms) if terms else "false") + ")"

    FCMP = {"lt": "<", "eq": "==", "le": "<=", "gt": ">", "ne": "!=", "ge": ">="}

    def fcmp(self, op, a, b):
        if op in ("lt", "eq", "le", "gt", "ge"):
            return f"({a} {self.FCMP[op]} {b})"
        if op == "ne":
            return f"({a} != {b} && !isnan({a}) && !isnan({b}))"
        if op == "num":
            return f"(!isnan({a}) && !isnan({b}))"
        if op == "nan":
            return f"(isnan({a}) || isnan({b}))"
        if op in ("ltu", "equ", "leu", "gtu", "geu"):
            return f"(!({a} {self.FCMP[{'ltu': 'ge', 'equ': 'ne', 'leu': 'gt', 'gtu': 'le', 'geu': 'lt'}[op]]} {b}) || isnan({a}) || isnan({b}))" \
                if op != "equ" else f"({a} == {b} || isnan({a}) || isnan({b}))"
        if op == "neu":
            return f"({a} != {b})"
        if op == "t":
            return "true"
        return "false"

    def emit_setp(self, i):
        m, s = i.mods, i.srcs
        bop = {"and": "&&", "or": "||", "xor": "!="}[m["bop"]]
        acc = self.p(i.psrc[0])
        if i.name in ("fsetp", "fset"):
            c = self.fcmp(m["cmp"], self.f(s[0]), self.f(s[1]))
        elif i.name == "hsetp2":
            # comparaison f16x2 par moitié : sans H_AND, P = moitié basse, Q = moitié haute ; avec
            # H_AND, P = (basse && haute), Q = son inverse (chacun combiné à l'accumulateur)
            a2, b2 = self.h2v(s[0], m.get("sa", 0)), self.h2v(s[1], m.get("sb", 0))
            cx = self.fcmp(m["cmp"], f"({a2}).x", f"({b2}).x")
            cy = self.fcmp(m["cmp"], f"({a2}).y", f"({b2}).y")
            if m.get("h_and"):
                out = f"{{ bool t = ({cx} && {cy}); "
                out += self.setp(i.pdst[0], f"(t {bop} {acc})") + " "
                if len(i.pdst) > 1 and i.pdst[1].n != PT:
                    out += self.setp(i.pdst[1], f"(!t {bop} {acc})")
                return out + " }"
            out = f"{{ bool tx = {cx}; bool ty = {cy}; "
            out += self.setp(i.pdst[0], f"(tx {bop} {acc})") + " "
            if len(i.pdst) > 1 and i.pdst[1].n != PT:
                out += self.setp(i.pdst[1], f"(ty {bop} {acc})")
            return out + " }"
        else:
            op = m["cmp"]
            if op in ("f", "t"):
                c = "false" if op == "f" else "true"
            else:
                if m["signed"]:
                    a, b = f"int({self.ui(s[0])})", f"int({self.ui(s[1])})"
                else:
                    a, b = self.ui(s[0]), self.ui(s[1])
                c = f"({a} {self.FCMP[op]} {b})"
                if m.get("ex"):
                    low = self.p(i.psrc[1])
                    eq = f"({a} == {b})"
                    if op in ("lt", "le", "gt", "ge"):
                        strict = {"lt": "<", "le": "<", "gt": ">", "ge": ">"}[op]
                        c = f"(({a} {strict} {b}) || ({eq} && {low}))"
                    elif op == "eq":
                        c = f"({eq} && {low})"
                    else:
                        c = f"(!{eq} || {low})"
        if i.name == "fset":
            val = "1.0" if m.get("bf") else None
            e = f"(({c}) {bop} {acc})"
            return self.setr(i.dst, f"({e} ? {'floatBitsToUint(1.0)' if val else '0xffffffffu'} : 0u)")
        out = f"{{ bool t = {c}; "
        out += self.setp(i.pdst[0], f"(t {bop} {acc})") + " "
        if len(i.pdst) > 1 and i.pdst[1].n != PT:
            out += self.setp(i.pdst[1], f"(!t {bop} {acc})")
        return out + " }"

    def emit_iadd3(self, i):
        s, m = i.srcs, i.mods
        a, b, c = self.ui(s[0]), self.ui(s[1]), self.ui(s[2])
        need_c = any(pd.n != PT for pd in i.pdst)
        cin = ""
        if m.get("x"):
            cin = f" + ({self.p(i.psrc[0])} ? 1u : 0u) + ({self.p(i.psrc[1])} ? 1u : 0u)"
        if not need_c:
            return self.setr(i.dst, f"({a} + {b} + {c}{cin})")
        # retenues : somme sur 64 bits
        self.uses.add("int64")
        ex = lambda o, v: f"(uint64_t(~{self.u(o)}) + 1ul)" if getattr(o, "neg", False) else f"uint64_t({v})"
        terms = [ex(s[0], self.u(s[0])), ex(s[1], self.u(s[1])), ex(s[2], self.u(s[2]))]
        sum_ = " + ".join(t for t in terms)
        if m.get("x"):
            sum_ += f" + uint64_t({self.p(i.psrc[0])} ? 1u : 0u) + uint64_t({self.p(i.psrc[1])} ? 1u : 0u)"
        out = f"{{ uint64_t t = {sum_}; {self.setr(i.dst, 'uint(t)')} "
        out += self.setp(i.pdst[0], "((t >> 32) & 1ul) != 0ul") + " "
        if len(i.pdst) > 1:
            out += self.setp(i.pdst[1], "((t >> 33) & 1ul) != 0ul")
        return out + " }"

    def emit_shf(self, i):
        s, m = i.srcs, i.mods
        self.uses.add("int64")
        lo, sh, hi = self.u(s[0]), self.u(s[1]), self.u(s[2])
        t = m["type"]
        is64 = t in ("s64", "u64")
        mask = 63 if is64 else 31
        shv = f"({sh} & {mask}u)" if m["wrap"] else f"min({sh}, {64 if is64 else 32}u)"
        v = f"(uint64_t({hi}) << 32 | uint64_t({lo}))"
        if m["right"]:
            if t == "s64" or t == "s32":
                src = f"int64_t({v})" if t == "s64" else f"int64_t(int({lo}))" if s[2].zero else f"int64_t({v})"
                r = f"uint64_t({src} >> {shv})"
            else:
                r = f"({v} >> {shv})"
        else:
            r = f"({v} << {shv})"
        res = f"uint({r} >> 32)" if m["hi"] else f"uint({r})"
        if not is64 and not m["right"] and not m["hi"]:
            res = f"({lo} << {shv})" if m["wrap"] else f"(({shv}) >= 32u ? 0u : {lo} << {shv})"
        return self.setr(i.dst, res)

    def h2v(self, o, swz):
        """Opérande f16x2 en vec2 (sélecteur : 0 H1_H0, 1 F32 diffusé, 2 H0_H0, 3 H1_H1)."""
        if isinstance(o, Imm):
            v = f"unpackHalf2x16(0x{o.v:08x}u)"
        elif isinstance(o, Reg) and o.zero:
            v = "vec2(0.0)"
        else:
            raw = self.u(o)
            v = {0: f"unpackHalf2x16({raw})", 1: f"vec2(uintBitsToFloat({raw}))",
                 2: f"unpackHalf2x16({raw}).xx", 3: f"unpackHalf2x16({raw}).yy"}[swz]
        if getattr(o, "abs", False):
            v = f"abs({v})"
        if getattr(o, "neg", False):
            v = f"(-{v})"
        return v

    def emit_h2(self, i):
        s, m = i.srcs, i.mods
        h = self.h2v
        a = h(s[0], m.get("sa", 0))
        b = h(s[1], m.get("sb", 0) if isinstance(s[1], Reg) and s[1] is not None else 0)
        if i.name == "hadd2":
            e = f"({a} + {b})"
        elif i.name == "hmul2":
            e = f"({a} * {b})"
        else:
            c = h(s[2], m.get("sc", 0))
            e = f"fma({a}, {b}, {c})"
        if m.get("sat"):
            e = f"clamp({e}, vec2(0.0), vec2(1.0))"
        if m.get("f32"):
            return self.setf(i.dst, f"({e}).x")
        return self.setr(i.dst, f"packHalf2x16({e})")

    SR = {0x21: "gl_LocalInvocationID.x", 0x22: "gl_LocalInvocationID.y", 0x23: "gl_LocalInvocationID.z",
          0x25: "gl_WorkGroupID.x", 0x26: "gl_WorkGroupID.y", 0x27: "gl_WorkGroupID.z"}

    def emit_sr(self, i):
        sr = i.mods["sr"]
        if sr == 0xff:
            out = self.setr(i.dst, "0u")
            if i.name == "cs2r" and i.mods.get("w64"):
                out += " " + self.setr(self.rn(i.dst, 1), "0u")
            return out
        if sr in self.SR:
            if self.kind != "comp":
                raise Unsupported(f"special register {sr:#x} outside compute")
            return self.setr(i.dst, f"{self.SR[sr]}")
        if sr == 0x00:
            self.uses.add("subgroup")
            return self.setr(i.dst, "gl_SubgroupInvocationID")
        if sr in (0x38, 0x39, 0x3a, 0x3b, 0x3c):
            self.uses.add("subgroup")
            nm = {0x38: "Eq", 0x39: "Lt", 0x3a: "Le", 0x3b: "Gt", 0x3c: "Ge"}[sr]
            return self.setr(i.dst, f"gl_Subgroup{nm}Mask.x")
        if sr in (0x50, 0x51):
            return self.setr(i.dst, "0u")
        if sr == 0x12:   # SR_Y_DIRECTION (signe de l'axe Y)
            return self.setf(i.dst, "1.0")
        self.warn.append(f"special register 0x{sr:x} → 0")
        return self.setr(i.dst, "0u")

    def emit_ldc(self, i):
        c, off = i.srcs
        t = i.mods["type"]
        base = c.off
        if c.ubank is not None:
            raise Unsupported("LDC bindless")
        idx = f"(({self.u(off)} + {base}u) >> 2)" if not off.zero else f"{base // 4}u"
        bank = c.bank
        if t == 4:
            return self.setr(i.dst, self.cb(bank, idx))
        if t in (5, 6):
            n = 2 if t == 5 else 4
            parts = [self.setr(self.rn(i.dst, k), self.cb(bank, f"({idx} + {k}u)")) for k in range(n)]
            return "{ " + " ".join(parts) + " }"
        if t in (0, 1, 2, 3):
            byteoff = f"({self.u(off)} + {base}u)" if not off.zero else f"{base}u"
            w = self.cb(bank, f"({byteoff} >> 2)")
            width = 8 if t < 2 else 16
            signed = t in (1, 3)
            return self.setr(i.dst, f"uint(bitfieldExtract({'int' if signed else ''}({w}), int(({byteoff} & 3u) * 8u), {width}))")
        raise Unsupported(f"ldc type {t}")

    def emit_attr(self, i):
        m = i.mods
        addr = m["addr"]
        if i.name == "ipa":
            if self.kind != "frag":
                raise Unsupported("ipa outside fragment")
            if not i.srcs[0].zero:
                raise Unsupported("ipa indirect")
            return self.setf(i.dst, self.attr_in(addr, flat=m["freq"] == 1))
        if not (i.srcs[0].zero if i.name == "ald" else i.srcs[1].zero):
            raise Unsupported(f"{i.name} indirect")
        n = m["n"]
        out = []
        for k in range(n):
            a = addr + 4 * k
            if i.name == "ald":
                if m["out"] or m["patch"]:
                    raise Unsupported("ald output/patch")
                out.append(self.setr(self.rn(i.dst, k), self.attr_in_u(a)))
            else:
                data = Reg(i.srcs[0].n + k if i.srcs[0].n != RZ else RZ)
                out.append(self.attr_out(a, self.u(data)))
        return " ".join(x for x in out if x)

    def attr_in_u(self, a):
        if a == 0x2fc:
            return "uint(gl_VertexIndex)"
        if a == 0x2f8:
            return "uint(gl_InstanceIndex)"
        if self.kind == "vert" and 0x80 <= a < 0x280:
            k, c = (a - 0x80) // 16, ((a - 0x80) % 16) // 4
            self.vin.add(k)
            return f"floatBitsToUint(vin{k}[{c}])"
        raise Unsupported(f"input attribute 0x{a:x} ({self.kind})")

    def imap_mode(self, k, c):
        """Mode d'interpolation de l'entrée générique k, composante c, d'après l'IMAP du SPH (PS) :
        0 inutilisée, 1 constante, 2 perspective, 3 linéaire à l'écran (perspective par défaut)."""
        sph = getattr(self.st, "sph", b"")
        if len(sph) < 0x38:
            return 2
        return (sph[0x18 + k] >> (2 * c)) & 3 or 2

    def attr_in(self, a, flat=False):
        if 0x70 <= a < 0x80:
            return f"gl_FragCoord[{(a - 0x70) // 4}]"
        if 0x80 <= a < 0x280:
            k, c = (a - 0x80) // 16, ((a - 0x80) % 16) // 4
            mode = 1 if flat else self.imap_mode(k, c)
            q = {1: "flat", 3: "noperspective"}.get(mode, "")
            if self.fin.get(k) != "flat":   # une composante constante impose flat au vecteur
                self.fin[k] = q
            # IPA d'une entrée en perspective : le matériel interpole attribut/w linéairement à l'écran,
            # le shader remultiplie ensuite par 1/gl_FragCoord.w (= w) ; l'entrée « smooth » de Vulkan
            # est déjà corrigée de la perspective, d'où le facteur gl_FragCoord.w (sinon les UV des
            # objets 3D sont multipliées par leur profondeur : motifs « tissés »)
            if mode == 2:
                return f"(fin{k}[{c}] * gl_FragCoord.w)"
            return f"fin{k}[{c}]"
        if a == 0x3fc:   # masque entier : transtypage exact (une constante flottante NaN peut être canonisée)
            return "uintBitsToFloat(gl_FrontFacing ? 0xffffffffu : 0u)"
        if a in (0x2e0, 0x2e4):
            return f"gl_PointCoord[{(a - 0x2e0) // 4}]"
        raise Unsupported(f"fragment attribute 0x{a:x}")

    def attr_out(self, a, val):
        fv = f"uintBitsToFloat({val})"
        if 0x70 <= a < 0x80:
            return f"gl_Position[{(a - 0x70) // 4}] = {fv};"
        if 0x80 <= a < 0x280:
            k, c = (a - 0x80) // 16, ((a - 0x80) % 16) // 4
            self.vout.add(k)
            return f"vout{k}[{c}] = {fv};"
        if a == 0x6c:
            return f"gl_PointSize = {fv};"
        if a == 0x64:
            self.uses.add("layer")
            return f"gl_Layer = int({val});"
        if a == 0x68:
            self.uses.add("viewport")
            return f"gl_ViewportIndex = int({val});"
        if 0x2c0 <= a < 0x2e0:
            if os.environ.get("SHADERC_NO_CLIP"):   # diagnostic : distances de découpe ignorées
                return ""
            self.uses.add("clip")
            return f"gl_ClipDistance[{(a - 0x2c0) // 4}] = {fv};"
        raise Unsupported(f"output attribute 0x{a:x}")

    def tex_bank(self, m):
        """Banque des poignées liées ; formes 0xf60/0xf6c (indice) : 2 en graphique, 0 en calcul."""
        b = m.get("tbank")
        return (0 if self.kind == "comp" else 2) if b is None else b

    def emit_tex(self, i):
        m = i.mods
        bindless = bool(m.get("bindless"))
        if bindless and i.name == "txq":
            raise Unsupported("txq bindless")
        dim = m.get("dim", "2d")
        if "ur" in m:
            if i.name == "txq":
                raise Unsupported("txq uniform handle")
            h = None
        else:
            h = None if bindless else self.cb(self.tex_bank(m), f"{m['toff_cb'] // 4}u")
            if not bindless:
                self.tex_handles.add((self.tex_bank(m), m["toff_cb"]))
        name = i.name
        if name == "txq":
            if m["query"] != 0:
                raise Unsupported("txq type")
            # type deduit des autres acces a la meme poignee (TEX ou image), 2D par defaut
            ty = self.handle_dims().get((self.tex_bank(m), m["toff_cb"]), "2d")
            T = {"1d": "1D", "2d": "2D", "3d": "3D", "cube": "Cube", "1darray": "1DArray", "2darray": "2DArray",
                 "cubearray": "CubeArray"}[ty]
            self.uses.add("tex_" + ty)
            lod = self.u(i.srcs[0])
            t = f"T{T.upper()}[nonuniformEXT(min({h} & 0xfffffu, 16383u))]"
            sz = f"textureSize({t}, int({lod}))"
            n = {"1d": 1, "2d": 2, "cube": 2, "1darray": 2}.get(ty, 3)
            v = {1: f"uvec3(uint({sz}), 1u, 1u)", 2: f"uvec3(uvec2({sz}), 1u)", 3: f"uvec3({sz})"}[n]
            e = f"uvec4({v}, uint(textureQueryLevels({t})))"
            return self.tex_out(i, e)
        arr = dim in ("1darray", "2darray", "cubearray")
        ncoord = {"1d": 1, "2d": 2, "3d": 3, "cube": 3, "1darray": 1, "2darray": 2, "cubearray": 3}[dim]
        comps = []
        if arr:
            comps.append(("arr", None))
        comps += [("c", k) for k in range(ncoord)]
        lodm = m.get("lod", "auto")
        if name in ("tex", "tld") and lodm in ("ll", "lb", "lb.lc", "lc"):
            comps.append(("lod", None))
        if m.get("aoffi") or (name == "tld4" and m.get("toff")):
            comps.append(("off", None))
        if m.get("dc"):
            comps.append(("dc", None))
        if bindless:
            # TEX bindless (0x361…) : poignée combinée en dernier opérande (établi sur le matériau
            # principal : 2D → Ra = u, v et Rb = poignée ; 3D → Ra = x, y et Rb = z, poignée)
            comps.append(("handle", None))
        na = (len(comps) + 1) // 2
        regs = []
        ra, rb = i.srcs[0], i.srcs[1]
        for k in range(len(comps)):
            if k < na:
                regs.append(self.u(Reg(ra.n + k if ra.n != RZ else RZ)))
            else:
                regs.append(self.u(Reg(rb.n + (k - na) if rb.n != RZ else RZ)))
        val = dict()
        coords = []
        for (kind, k), r in zip(comps, regs):
            if kind == "c":
                coords.append(r)
            else:
                val[kind] = r
        is_int = name == "tld"
        if is_int:
            cvec = [f"int({c})" for c in coords]
        else:
            cvec = [f"uintBitsToFloat({c})" for c in coords]
        ty = {"1d": "1D", "2d": "2D", "3d": "3D", "cube": "Cube", "1darray": "1DArray", "2darray": "2DArray",
              "cubearray": "CubeArray"}[dim]
        self.uses.add("tex_" + dim)
        if bindless:
            h = val["handle"]
            self.uses.add("bindless")
        # identifiant borné à la taille des tableaux du jeu 2 (TEX_MAX = 16384 dans vk_render.cpp) et
        # combinaison texture + échantillonneur marquée non uniforme : sans NonUniform sur le résultat
        # d'OpSampledImage, le pilote NVIDIA lit la poignée dans un seul fil, éventuellement inactif
        # (branche divergente) au registre quelconque → en-tête de texture invalide, faute de page GPU
        tid, samp = self.tex_ids(m, ty, h)
        sampler = f"nonuniformEXT(sampler{ty}{'Shadow' if m.get('dc') else ''}({tid}, {samp}))"
        gather_off = None
        if "off" in val:
            # Décalages dynamiques (registre) : GLSL n'accepte que des constantes pour textureOffset et
            # texelFetchOffset. TLD4.AOFFI : textureGatherOffset (décalage dynamique permis, 6 bits signés
            # aux bits 0 et 8). TEX / TLD.AOFFI (4 bits signés par composante) : ajoutés aux coordonnées,
            # exact pour texelFetch, en texels du niveau lu pour texture (niveau 0 hors LOD explicite).
            ov = val["off"]
            if name == "tld4":
                if m.get("toff") != 1 or dim not in ("2d", "2darray"):
                    raise Unsupported("tld4 per-texel offset (PTP)")
                gather_off = f"ivec2(bitfieldExtract(int({ov}), 0, 6), bitfieldExtract(int({ov}), 8, 6))"
            else:
                if dim in ("cube", "cubearray"):
                    raise Unsupported("aoffi cube")
                offs = [f"bitfieldExtract(int({ov}), {4 * k}, 4)" for k in range(ncoord)]
                if is_int:
                    cvec = [f"({c} + {o})" for c, o in zip(cvec, offs)]
                else:
                    if lodm == "ll":
                        lv = f"int(uintBitsToFloat({val['lod']}))"
                    elif lodm in ("auto", "lb") and self.kind == "frag":
                        q = cvec[0] if ncoord == 1 else f"vec{ncoord}({', '.join(cvec)})"
                        lv = f"int(textureQueryLod({sampler}, {q}).x)"
                    else:
                        lv = "0"
                    sz = f"textureSize({tid}, {lv})"
                    sw = [""] if (ncoord == 1 and not arr) else [".x", ".y", ".z"]
                    cvec = [f"({c} + float({o}) / float({sz}{sw[k] if len(sw) > 1 else ''}))"
                            for k, (c, o) in enumerate(zip(cvec, offs))]
        if arr:
            cvec.append(f"float({val['arr']})" if not is_int else f"int({val['arr']})")
        n = len(cvec)
        vt = ("ivec" if is_int else "vec") + (str(n) if n > 1 else "")
        if n == 1:
            vt = "int" if is_int else "float"
        cexpr = f"{vt}({', '.join(cvec)})"
        if name == "tld":
            lod = f"int({val['lod']})" if "lod" in val else "0"
            e = f"floatBitsToUint(texelFetch({tid}, {cexpr}, {lod}))"
        elif name == "tld4":
            if m.get("dc"):   # textureGather avec comparaison : 4 résultats 0/1
                if gather_off:
                    e = f"floatBitsToUint(textureGatherOffset({sampler}, {cexpr}, uintBitsToFloat({val['dc']}), {gather_off}))"
                else:
                    e = f"floatBitsToUint(textureGather({sampler}, {cexpr}, uintBitsToFloat({val['dc']})))"
            elif gather_off:
                e = f"floatBitsToUint(textureGatherOffset({sampler}, {cexpr}, {gather_off}, {m.get('comp', 0)}))"
            else:
                e = f"floatBitsToUint(textureGather({sampler}, {cexpr}, {m.get('comp', 0)}))"
        elif m.get("dc"):
            # TEX.DC : comparaison avec la profondeur de référence (dernier opérande) ; résultat dans R
            ref = f"uintBitsToFloat({val['dc']})"
            nn = n + 1
            sv = f"vec{nn}({', '.join(cvec)}, {ref})" if nn <= 4 else None
            if sv is None:   # cube array + référence : la référence est un argument séparé
                if lodm not in ("auto",) or self.kind != "frag":
                    raise Unsupported("tex dc cube array lod")
                r = f"texture({sampler}, {cexpr}, {ref})"
            elif lodm == "auto" and self.kind == "frag":
                r = f"texture({sampler}, {sv})"
            elif lodm in ("auto", "lz") and ty == "2D":
                r = f"textureLod({sampler}, {sv}, 0.0)"
            elif lodm in ("auto", "lz"):
                g = {"1D": "0.0", "1DArray": "0.0", "2DArray": "vec2(0.0)", "Cube": "vec3(0.0)"}.get(ty)
                if g is None:
                    raise Unsupported(f"tex dc {ty} lod")
                r = f"textureGrad({sampler}, {sv}, {g}, {g})"
            elif lodm == "ll" and ty in ("1D", "2D"):
                r = f"textureLod({sampler}, {sv}, uintBitsToFloat({val['lod']}))"
            elif lodm == "lb" and self.kind == "frag":
                r = f"texture({sampler}, {sv}, uintBitsToFloat({val['lod']}))"
            else:
                raise Unsupported(f"tex dc lod {lodm}")
            e = f"uvec4(floatBitsToUint({r}), 0u, 0u, 0u)"
        else:
            if lodm == "lz":
                e = f"textureLod({sampler}, {cexpr}, 0.0)"
            elif lodm == "ll":
                e = f"textureLod({sampler}, {cexpr}, uintBitsToFloat({val['lod']}))"
            elif lodm == "lb":
                e = f"texture({sampler}, {cexpr}, uintBitsToFloat({val['lod']}))"
            elif lodm == "auto":
                if self.kind != "frag":
                    e = f"textureLod({sampler}, {cexpr}, 0.0)"
                else:
                    e = f"texture({sampler}, {cexpr})"
            else:
                raise Unsupported(f"tex lod {lodm}")
            e = f"floatBitsToUint({e})"
        return self.tex_out(i, e)

    def emit_txd(self, i):
        """TXD : lecture avec dérivées explicites (textureGrad).

        Opérandes établis sur le matériau principal du jeu : Ra = coordonnées puis indice de couche
        (tableaux), Rb = dérivées entrelacées par coordonnée (d/dx, d/dy de la 1re, puis de la 2e…)."""
        m = i.mods
        if m.get("bindless"):
            raise Unsupported("texture bindless")
        dim = m.get("dim", "2d")
        if dim not in ("1d", "2d", "3d", "cube", "1darray", "2darray"):
            raise Unsupported(f"txd {dim}")
        if m.get("aoffi") or m.get("dc"):
            raise Unsupported("txd aoffi/dc")
        h = None if "ur" in m else self.cb(self.tex_bank(m), f"{m['toff_cb'] // 4}u")
        if "ur" not in m:
            self.tex_handles.add((self.tex_bank(m), m["toff_cb"]))
        ncoord = {"1d": 1, "2d": 2, "3d": 3, "cube": 3, "1darray": 1, "2darray": 2}[dim]
        ra, rb = i.srcs[0], i.srcs[1]

        def reg(base, k):
            return self.u(Reg(base.n + k if base.n != RZ else RZ))

        coords = [f"uintBitsToFloat({reg(ra, k)})" for k in range(ncoord)]
        if dim.endswith("array"):
            coords.append(f"float({reg(ra, ncoord)})")
        dx = [f"uintBitsToFloat({reg(rb, 2 * k)})" for k in range(ncoord)]
        dy = [f"uintBitsToFloat({reg(rb, 2 * k + 1)})" for k in range(ncoord)]

        def vec(xs):
            return xs[0] if len(xs) == 1 else f"vec{len(xs)}({', '.join(xs)})"

        ty = {"1d": "1D", "2d": "2D", "3d": "3D", "cube": "Cube", "1darray": "1DArray", "2darray": "2DArray"}[dim]
        self.uses.add("tex_" + dim)
        tid, samp = self.tex_ids(m, ty, h)
        e = f"floatBitsToUint(textureGrad(nonuniformEXT(sampler{ty}({tid}, {samp})), {vec(coords)}, {vec(dx)}, {vec(dy)}))"
        return self.tex_out(i, e)

    def tex_ids(self, m, ty, h):
        """Texture et échantillonneur d'une lecture : poignée combinée h (texture | échantillonneur << 20)
        ou, forme 0xf60/0xf6c, paire de registres uniformes URn (texture) / URn+1 (échantillonneur)."""
        if "ur" in m:
            n = m["ur"]
            ut = "0u" if n >= 63 else f"UR{n}"
            us = "0u" if n + 1 >= 63 else f"UR{n + 1}"
            return (f"T{ty.upper()}[nonuniformEXT(min({ut} & 0xfffffu, 16383u))]",
                    f"SMP[nonuniformEXT({us} & 0xfffu)]")
        return (f"T{ty.upper()}[nonuniformEXT(min({h} & 0xfffffu, 16383u))]", f"SMP[nonuniformEXT({h} >> 20)]")

    IMG_DIMS = {0: ("1D", 1, "int"), 2: ("1DArray", 2, "ivec2"), 3: ("2D", 2, "ivec2"),
                4: ("2DArray", 3, "ivec3"), 5: ("3D", 3, "ivec3")}

    def emit_surf(self, i):
        """SULD.P / SUST.P (format de l'image) : imageLoad / imageStore sans format déclaré."""
        m = i.mods
        if m.get("bindless"):
            raise Unsupported("image bindless")
        if not m.get("fmt"):
            raise Unsupported(f"{i.name}.D (raw access)")
        if m["dim"] not in self.IMG_DIMS:
            raise Unsupported(f"image dim {m['dim']}")
        ty, nc, ct = self.IMG_DIMS[m["dim"]]
        self.uses.add("img_" + ty.lower())
        self.img_handles.add((m["tbank"], m["toff_cb"], ty.lower()))
        h = self.cb(m["tbank"], f"{m['toff_cb'] // 4}u")
        ra = i.srcs[0]
        cs = [f"int({self.u(Reg(ra.n + k if ra.n != RZ else RZ))})" for k in range(nc)]
        coord = cs[0] if nc == 1 else f"{ct}({', '.join(cs)})"
        if i.name in ("suatom", "sured"):
            # atomiques d'image 32 bits non signés : déclaration r32ui du même tableau (jeu 2, même liaison)
            if m["type"] != 0:
                raise Unsupported(f"{i.name} type {m['type']}")
            fn = {0: "imageAtomicAdd", 1: "imageAtomicMin", 2: "imageAtomicMax", 5: "imageAtomicAnd", 6: "imageAtomicOr",
                  7: "imageAtomicXor", 8: "imageAtomicExchange"}.get(m["op"])
            if fn is None:
                raise Unsupported(f"{i.name} op {m['op']}")
            self.uses.add("imgu_" + ty.lower())
            img = f"IMGU{ty.upper()}[nonuniformEXT(min({h} & 0xfffffu, 16383u))]"
            expr = f"{fn}({img}, {coord}, {self.u(i.srcs[1])})"
            if i.dst is None or i.dst.zero:
                return expr + ";"
            return self.setr(i.dst, expr)
        img = f"IMG{ty.upper()}[nonuniformEXT(min({h} & 0xfffffu, 16383u))]"
        mask = m["mask"]
        if i.name == "sust":
            rb = i.srcs[1]
            vals, k = [], 0
            for c in range(4):
                if mask >> c & 1:
                    vals.append(f"uintBitsToFloat({self.u(Reg(rb.n + k if rb.n != RZ else RZ))})")
                    k += 1
                else:
                    vals.append("0.0")
            return f"imageStore({img}, {coord}, vec4({', '.join(vals)}));"
        out = [f"{{ uvec4 t = floatBitsToUint(imageLoad({img}, {coord}));"]
        k = 0
        for c in range(4):
            if mask >> c & 1:
                if i.dst.n + k != RZ:
                    out.append(self.setr(Reg(i.dst.n + k), f"t[{c}]"))
                k += 1
        out.append("}")
        return " ".join(out)

    def handle_dims(self):
        """(banque, deplacement) d'une poignee -> type de texture, d'apres les TEX et acces images."""
        if not hasattr(self, "_hdims"):
            IMGT = {0: "1d", 2: "1darray", 3: "2d", 4: "2darray", 5: "3d"}
            d = {}
            for ins in self.insns:
                mm = ins.mods
                if "toff_cb" not in mm or "tbank" not in mm:
                    continue
                if ins.name in ("suld", "sust", "suatom", "sured") and mm.get("dim") in IMGT:
                    d[(self.tex_bank(mm), mm["toff_cb"])] = IMGT[mm["dim"]]
                elif ins.name in ("tex", "tld", "tld4", "txd", "tmml") and mm.get("dim") in (
                        "1d", "2d", "3d", "cube", "1darray", "2darray", "cubearray"):
                    d.setdefault((self.tex_bank(mm), mm["toff_cb"]), mm["dim"])
            self._hdims = d
        return self._hdims

    def tex_out(self, i, e):
        mask = i.mods["mask"]
        d0, d1 = i.dst
        if i.mods.get("f16"):
            # .F16 : composantes activées deux par deux en demi-flottants, (c0 bas, c1 haut) dans Rd puis (c2, c3)
            # dans Rd2 (PTX tex.v2.f16x2 → [Rd, Rd2]) ; moitié haute sans composante : 0
            if i.name not in ("tex", "tld"):
                raise Unsupported(f"{i.name}.F16")
            comps = [c for c in range(4) if mask >> c & 1]
            f = lambda c: f"uintBitsToFloat(t[{c}])"
            out = [f"{{ uvec4 t = {e};"]
            for k, dr in enumerate((d0, d1)):
                pair = comps[2 * k:2 * k + 2]
                if pair and dr.n != RZ:
                    v = f"vec2({f(pair[0])}, {f(pair[1]) if len(pair) > 1 else '0.0'})"
                    out.append(self.setr(dr, f"packHalf2x16({v})"))
            out.append("}")
            return " ".join(out)
        dsts = [d0.n, d0.n + 1 if d0.n != RZ else RZ, d1.n, d1.n + 1 if d1.n != RZ else RZ]
        out = [f"{{ uvec4 t = {e};"]
        k = 0
        for c in range(4):
            if mask >> c & 1:
                if k < 4 and dsts[k] != RZ:
                    out.append(self.setr(Reg(dsts[k]), f"t[{c}]"))
                k += 1
        out.append("}")
        return " ".join(out)

    # ---------------------------------------------------------------- mémoire globale
    # Les adresses invitées (pointeurs hôtes des pools NVN) viennent des descripteurs de tampons (16 o) de la
    # banque 0, ou de données du jeu (pointeurs rangés dans un tampon uniforme : lumière du brouillard). Chaque
    # accès passe par gxl() : la table GTAB écrite par le runtime en fin de banque 0 (vk_render.cpp,
    # dispatch) donne le nombre de pools importés puis, par pool, base invitée, adresse Vulkan (pointeur de
    # tampon) et taille ; hors des pools, lecture 0 et écriture ignorée (jamais de faute GPU).
    GTAB_WORD = 0x600 // 4

    def gaddr(self, i):
        ra, off = i.srcs[0], i.srcs[1].v
        if off >= 1 << 31:
            off -= 1 << 32
        if i.mods.get("e", 1):
            base = "0ul" if ra.zero else f"packUint2x32(uvec2({self.u(ra)}, {self.u(self.rn(ra, 1))}))"
        else:
            base = f"uint64_t({self.u(ra)})"
        ur = i.srcs[2] if i.opc == 0x981 else i.srcs[3] if i.opc == 0x986 else None
        if ur is not None and not ur.zero:
            base = f"({base} + packUint2x32(uvec2({self.ur(ur.n)}, {self.ur(ur.n + 1)})))"
        if off:
            base = f"({base} + uint64_t(int64_t({off})))"
        return base

    def emit_global(self, i):
        # Hors calcul : lectures seules, en fragments (color_plot_draw.frag, f979ce45bdf2392a, lit son tampon de
        # stockage par LDG à l'adresse c[0][0x5b0]) ; la banque 0 de l'étage porte alors le descripteur et la table
        # GTAB, comme en calcul (vk_render.cpp, banque0_graphique). Écritures et atomiques : non traduites.
        if self.kind != "comp" and not (self.kind == "frag" and i.name == "ldg"):
            raise Unsupported(f"global memory outside compute ({i.name})")
        self.uses.update(("int64", "global"))
        self.banks.add(0)
        n, m = i.name, i.mods
        a = self.gaddr(i)
        if n in ("ldg", "stg"):
            t = m.get("type", 4)
            if n == "ldg":
                if t <= 3:
                    w = 8 if t <= 1 else 16
                    v = f"gld{w}({a})"
                    if t in (1, 3):   # signés : extension de signe
                        v = f"uint(bitfieldExtract(int({v}), 0, {w}))"
                    return self.setr(i.dst, v)
                cnt = {4: 1, 5: 2, 6: 4, 7: 4}[t]
                if cnt == 1:
                    return self.setr(i.dst, f"gld({a})")
                out = ["{ uint64_t ga = " + a + ";"]
                out += [self.setr(self.rn(i.dst, k), f"gld(ga + {4 * k}ul)") for k in range(cnt)]
                return " ".join(out) + " }"
            data = i.srcs[2]
            if t <= 3:
                return f"gst{8 if t <= 1 else 16}({a}, {self.u(data)});"
            cnt = {4: 1, 5: 2, 6: 4, 7: 4}[t]
            out = ["{ uint64_t ga = " + a + ";"]
            out += [f"gst(ga + {4 * k}ul, {self.u(self.rn(data, k))});" for k in range(cnt)]
            return " ".join(out) + " }"
        # ATOMG / RED : 32 bits (types 0 non signé, 1 signé), addition flottante (type 3)
        self.uses.add("gatomic")
        op = m["op"]
        t = m.get("type", 0)
        b = self.u(i.srcs[2])
        if op == "cas":
            expr = f"gcas({a}, {b}, {self.u(i.srcs[3])})"
        else:
            if t == 3 and op == 0:
                fn = "gaddf"
            elif t in (0, 1):
                fn = {0: "gadd", 1: "gmin", 2: "gmax", 3: "ginc", 4: "gdec", 5: "gand", 6: "gor", 7: "gxor",
                      8: "gexch"}.get(op)
                if fn is None:
                    raise Unsupported(f"{n} op {op}")
                if t == 1 and op in (1, 2):
                    fn += "s"
            else:
                raise Unsupported(f"{n} type {t}")
            expr = f"{fn}({a}, {b})"
        if n == "red" or i.dst is None or i.dst.zero:
            return f"{expr};"
        return self.setr(i.dst, expr)

    def global_helpers(self):
        cb = f"cb{self.sidx}_0.d"
        coh = "coherent " if ("gatomic" in self.uses or "membar" in self.uses) else ""
        g = self.GTAB_WORD
        L = [f"layout(buffer_reference, std430, buffer_reference_align = 4) {coh}buffer GW {{ uint v; }};",
             f"layout(buffer_reference, std430, buffer_reference_align = 4) {coh}buffer GI {{ int v; }};",
             # dernier pool trouvé essayé d'abord (les accès d'un shader visent presque tous le même pool)
             "uint gxl_last = 0u;",
             "uint64_t gxl_try(uint k, uint64_t a) {",
             f"  uint b = {g + 4}u + 8u * k;",
             f"  uint64_t o = a - packUint2x32(uvec2({cb}[b], {cb}[b + 1u]));",
             f"  uint64_t sz = packUint2x32(uvec2({cb}[b + 4u], {cb}[b + 5u]));",
             f"  return (o < sz && sz - o >= 4ul) ? packUint2x32(uvec2({cb}[b + 2u], {cb}[b + 3u])) + o : 0ul;",
             "}",
             # Adresse absente de la table : notée dans le tampon de retour du runtime (son adresse GPU est dans la table,
             # mots g + 2 et g + 3 ; 0 : pas de tampon), qui importera sa fenêtre pour les calculs suivants
             # (vk_render.cpp, glob_retour). Depuis la 1.0.3, la mémoire du jeu n'est plus importée en entier.
             "layout(buffer_reference, std430, buffer_reference_align = 8) buffer GFB { uint n; uint pad; uint64_t a[255]; };",
             "void gxl_miss(uint64_t a) {",
             f"  uint64_t f = packUint2x32(uvec2({cb}[{g + 2}u], {cb}[{g + 3}u]));",
             "  if (f == 0ul || a < 0x10000ul) return;",
             "  GFB fb = GFB(f);",
             "  if (fb.n >= 255u) return;",
             "  uint i = atomicAdd(fb.n, 1u);",
             "  if (i < 255u) fb.a[i] = a;",
             "}",
             "uint64_t gxl(uint64_t a) {",
             f"  uint n = min({cb}[{g}u], 78u);",
             "  if (gxl_last < n) { uint64_t d = gxl_try(gxl_last, a); if (d != 0ul) return d; }",
             "  for (uint k = 0u; k < n; k++) {",
             "    uint64_t d = gxl_try(k, a);",
             "    if (d != 0ul) { gxl_last = k; return d; }",
             "  }",
             "  gxl_miss(a);",
             "  return 0ul;",
             "}",
             "uint gld(uint64_t a) { uint64_t d = gxl(a & ~3ul); return d != 0ul ? GW(d).v : 0u; }",
             "void gst(uint64_t a, uint v) { uint64_t d = gxl(a & ~3ul); if (d != 0ul) GW(d).v = v; }",
             "uint gld8(uint64_t a) { return (gld(a) >> (uint(a & 3ul) * 8u)) & 0xffu; }",
             "uint gld16(uint64_t a) { return (gld(a) >> (uint(a & 2ul) * 8u)) & 0xffffu; }",
             # écritures partielles : deux opérations atomiques sur le mot (les octets voisins restent intacts)
             "void gst8(uint64_t a, uint v) { uint64_t d = gxl(a & ~3ul); if (d != 0ul) { uint s = uint(a & 3ul) * 8u;"
             " atomicAnd(GW(d).v, ~(0xffu << s)); atomicOr(GW(d).v, (v & 0xffu) << s); } }",
             "void gst16(uint64_t a, uint v) { uint64_t d = gxl(a & ~3ul); if (d != 0ul) { uint s = uint(a & 2ul) * 8u;"
             " atomicAnd(GW(d).v, ~(0xffffu << s)); atomicOr(GW(d).v, (v & 0xffffu) << s); } }"]
        if "gatomic" in self.uses:
            for fn, body in (("gadd", "atomicAdd(GW(d).v, v)"), ("gmin", "atomicMin(GW(d).v, v)"),
                             ("gmax", "atomicMax(GW(d).v, v)"), ("gand", "atomicAnd(GW(d).v, v)"),
                             ("gor", "atomicOr(GW(d).v, v)"), ("gxor", "atomicXor(GW(d).v, v)"),
                             ("gexch", "atomicExchange(GW(d).v, v)"), ("gmins", "uint(atomicMin(GI(d).v, int(v)))"),
                             ("gmaxs", "uint(atomicMax(GI(d).v, int(v)))")):
                L.append(f"uint {fn}(uint64_t a, uint v) {{ uint64_t d = gxl(a & ~3ul); return d != 0ul ? {body} : 0u; }}")
            L.append("uint gcas(uint64_t a, uint c, uint v) { uint64_t d = gxl(a & ~3ul);"
                     " return d != 0ul ? atomicCompSwap(GW(d).v, c, v) : 0u; }")
            # INC / DEC (sémantique CUDA) et addition flottante : boucles de comparaison-échange
            for fn, new in (("ginc", "o >= v ? 0u : o + 1u"), ("gdec", "(o == 0u || o > v) ? v : o - 1u"),
                            ("gaddf", "floatBitsToUint(uintBitsToFloat(o) + uintBitsToFloat(v))")):
                L.append(f"uint {fn}(uint64_t a, uint v) {{ uint64_t d = gxl(a & ~3ul); if (d == 0ul) return 0u;"
                         f" uint o = GW(d).v; for (uint n = 0u; n < 1024u; n++) {{ uint e = o;"
                         f" o = atomicCompSwap(GW(d).v, e, {new}); if (o == e) break; }} return o; }}")
        return L

    def saddr(self, i):
        """Adresse en octets d'un LDS / STS : [Ra.Xn + URb + imm]. L'échelle n (1, 4, 8, 16, bits 78-79) multiplie
        le registre : le compilateur indexe ainsi les tableaux partagés par le numéro du fil (ex. [R28.X8] pour des
        paires de flottants). Elle était ignorée : les flous de l'occlusion ambiante (SSAO) mélangeaient les
        valeurs de leurs voisins et l'image se couvrait d'un damier (entraînement, 2026-11-09)."""
        base, off = i.srcs[0], i.srcs[1].v
        parts = []
        if not base.zero:
            sc = i.mods.get("scale", 1)
            parts.append(f"{self.u(base)} * {sc}u" if sc != 1 else self.u(base))
        if i.name == "sts" and i.opc == 0x988 and len(i.srcs) > 3 and not i.srcs[3].zero:
            parts.append(self.ur(i.srcs[3].n))
        if not parts:
            return f"{off & 0xffffff}u"
        if off & 0xffffffff:
            parts.append(f"{off & 0xffffffff}u")
        return "(" + " + ".join(parts) + ")"

    def emit_mem(self, i):
        m = i.mods
        n = i.name
        if n in ("ldg", "stg", "atomg", "red"):
            return self.emit_global(i)
        t = m.get("type", 4)
        base, off = i.srcs[0], i.srcs[1].v
        if n in ("lds", "sts") and t <= 3:   # partage, 8 / 16 bits (U8, S8, U16, S16) : dans le mot de 32 bits
            self.uses.add("shared")
            w, mask = (8, "0xffu") if t <= 1 else (16, "0xffffu")
            al = "3u" if w == 8 else "2u"
            byte = self.saddr(i)
            if n == "lds":
                v = f"((SMEM[{byte} >> 2u] >> (({byte} & {al}) * 8u)) & {mask})"
                if t in (1, 3):
                    v = f"uint(bitfieldExtract(int({v}), 0, {w}))"
                return self.setr(i.dst, v)
            data = self.u(i.srcs[2])
            return (f"{{ uint sa = {byte}; uint s = (sa & {al}) * 8u; atomicAnd(SMEM[sa >> 2u], ~({mask} << s));"
                    f" atomicOr(SMEM[sa >> 2u], ({data} & {mask}) << s); }}")
        if n == "atoms" and t in (0, 1):   # ATOMS 32 bits : type 0 non signé, 1 signé
            t = 4
            if m.get("type") == 1 and m["op"] in (1, 2):   # min / max signés : boucle de comparaison-échange
                self.uses.add("shared")
                addr = f"(({self.u(base)} + {off & 0xffffffff}u) >> 2)" if not base.zero else f"{(off & 0xffffff) >> 2}u"
                fn = "min" if m["op"] == 1 else "max"
                v = self.u(i.srcs[2])
                return (f"{{ uint sa = {addr}; uint o = SMEM[sa]; for (uint n = 0u; n < 1024u; n++) {{ uint e = o;"
                        f" o = atomicCompSwap(SMEM[sa], e, uint({fn}(int(e), int({v})))); if (o == e) break; }} "
                        f"{self.setr(i.dst, 'o')} }}")
        if t not in (4, 5, 6):
            raise Unsupported(f"{n} type {t}")
        cnt = {4: 1, 5: 2, 6: 4}[t]
        if n in ("lds", "sts", "atoms"):
            self.uses.add("shared")
            if n == "atoms":   # pas d'échelle observée sur ATOMS dans ce jeu
                addr = f"(({self.u(base)} + {off & 0xffffffff}u) >> 2)" if not base.zero else f"{(off & 0xffffff) >> 2}u"
            else:
                sa = self.saddr(i)   # constante (sans registre) ou expression entre parenthèses
                addr = f"({sa} >> 2)" if sa.startswith("(") else f"{(off & 0xffffff) >> 2}u"
            if n == "lds":
                return "{ " + " ".join(self.setr(self.rn(i.dst, k), f"SMEM[{addr} + {k}u]") for k in range(cnt)) + " }"
            if n == "sts":
                data = i.srcs[2]
                return "{ " + " ".join(f"SMEM[{addr} + {k}u] = {self.u(Reg(data.n + k if data.n != RZ else RZ))};" for k in range(cnt)) + " }"
            op = m["op"]
            if op == "cas":
                return self.setr(i.dst, f"atomicCompSwap(SMEM[{addr}], {self.u(i.srcs[2])}, {self.u(i.srcs[3])})")
            fn = {0: "atomicAdd", 1: "atomicMin", 2: "atomicMax", 5: "atomicAnd", 6: "atomicOr", 7: "atomicXor",
                  8: "atomicExchange"}.get(op)
            if fn is None:
                raise Unsupported(f"atoms op {op}")
            return self.setr(i.dst, f"{fn}(SMEM[{addr}], {self.u(i.srcs[2])})")
        if n in ("ldl", "stl"):
            self.uses.add("local")
            addr = f"(({self.u(base)} + {off & 0xffffffff}u) >> 2)" if not base.zero else f"{(off & 0xffffff) >> 2}u"
            if n == "ldl":
                return "{ " + " ".join(self.setr(self.rn(i.dst, k), f"LMEM[{addr} + {k}u]") for k in range(cnt)) + " }"
            data = i.srcs[2]
            return "{ " + " ".join(f"LMEM[{addr} + {k}u] = {self.u(Reg(data.n + k if data.n != RZ else RZ))};" for k in range(cnt)) + " }"
        raise Unsupported(f"global memory ({n})")

    # ---------------------------------------------------------------- programme
    def translate(self):
        targets = {0}
        calls = False
        for i in self.insns:
            if i.target is not None and i.name in ("bra", "call") and i.target < len(self.insns) * 16:
                targets.add(i.target)
            if i.name == "call":
                calls = True
                targets.add(i.pc + 16)
            if i.name == "bra" and (i.mods or {}).get("noinc"):
                targets.add(i.pc + 16)   # adresse de retour possible d'un RET.REL.NODEC
            if i.name == "brx":
                # cibles possibles : adresses d'instruction présentes dans la banque 1 (tables de saut),
                # décalées de la base ; sans banque 1 vidée, la cible n'est pas connue
                cb1 = getattr(self.st, "cb1", b"")
                if not cb1:
                    raise Unsupported("brx without bank 1")
                words = struct.unpack_from(f"<{len(cb1) // 4}I", cb1)
                base = i.mods["base"]
                for v in words:
                    t = base + v
                    if t % 16 == 0 and 0 <= t < len(self.insns) * 16:
                        targets.add(t)
        body = []
        # HMMA partagé : au-delà de HMMA_EN_LIGNE instructions, un seul exemplaire de l'émulation, en tête de la chaîne
        # de blocs (bloc HMMA_PC) : chaque site range ses opérandes (HA, HB, HC), son adresse de retour (hret, pc + 8,
        # jamais une adresse d'instruction) et y saute ; le bloc revient dans la même passe. Copié en ligne 384 fois
        # (réseau NDSR 059b307b6448e8a1), il faisait planter le compilateur du pilote de NVIDIA.
        # variante cœurs tenseurs : chaque site est court, toujours en ligne (pas de passes de répartition)
        self.hmma_partage = not getattr(self, "coopmat", False) and sum(1 for i in self.insns if i.name == "hmma") > HMMA_EN_LIGNE
        if self.hmma_partage:
            self.uses.update(("subgroup", "tensor", "hmma"))
            body += [f"    case {HMMA_PC}u:", "      HD = hmma16816f16(HA, HB, HC); pc = hret; continue;"]
        for i in self.insns:
            if i.pc in targets:
                body.append(f"    case {i.pc}u:")
            g = i.guard
            if g.n == PT and g.neg:
                continue
            if i.name == "hmma" and self.hmma_partage:
                ra, rb, rc = i.srcs
                regs = lambda r, k: ", ".join(self.u(self.rn(r, j)) for j in range(k))
                body.append(f"      HA = uvec4({regs(ra, 4)}); HB = uvec2({regs(rb, 2)}); HC = uvec2({regs(rc, 2)});"
                            f" hret = {i.pc + 8}u; pc = {HMMA_PC}u; continue;  // hmma {i.pc:x}")
                body.append(f"    case {i.pc + 8}u:")
                body.append("      " + " ".join(x for x in (self.setr(i.dst, "HD.x"), self.setr(self.rn(i.dst, 1), "HD.y")) if x))
                continue
            if i.name == "bra":
                tgt = i.target
                if tgt == i.pc:
                    code = "return;"  # boucle d'arrêt après EXIT
                else:
                    code = f"pc = {tgt}u; continue;"
                cond = i.psrc[0] if i.psrc else Pred(PT)
                conds = [c for c in (g, cond) if not c.true]
                if conds:
                    code = f"if ({' && '.join(self.p(c) for c in conds)}) {{ {code} }}"
                body.append("      " + code)
                continue
            if i.name in ("brx", "retr"):   # saut indirect : table de saut (BRX) ou adresse de retour (RET.NODEC)
                code = f"pc = {i.mods['base'] & 0xffffffff}u + {self.u(i.srcs[0])}; continue;"
                cond = i.psrc[0] if i.psrc else Pred(PT)
                conds = [c for c in (g, cond) if not c.true]
                if conds:
                    code = f"if ({' && '.join(self.p(c) for c in conds)}) {{ {code} }}"
                body.append("      " + code + f"  // {i.name} {i.pc:x}")
                continue
            if i.name == "call":
                code = f"cstack[csp++] = {i.pc + 16}u; pc = {i.target}u; continue;"
            elif i.name == "ret":
                code = "pc = cstack[--csp]; continue;"
            elif i.name == "exit":
                code = self.exit_code()
                cond = i.psrc[0] if i.psrc else Pred(PT)
                if not cond.true:
                    code = f"if ({self.p(cond)}) {{ {code} }}"
            else:
                code = self.emit(i)
            if not code:
                continue
            if not g.true:
                code = f"if ({self.p(g)}) {{ {code} }}"
            body.append("      " + code + f"  // {i.pc:x}")
        return self.assemble(body, calls)

    def exit_code(self):
        if self.kind != "frag":
            return "return;"
        omap = struct.unpack_from("<I", self.st.sph, 0x48)[0]
        omap2 = struct.unpack_from("<I", self.st.sph, 0x4c)[0]
        reg = 0
        parts = []
        for rt in range(8):
            mk = (omap >> (4 * rt)) & 0xf
            if not mk:
                continue
            comps = []
            for c in range(4):
                if mk >> c & 1:
                    comps.append(f"uintBitsToFloat({self.r(reg)})")
                    reg += 1
                else:
                    comps.append("0.0")
            parts.append(f"fout{rt} = vec4({', '.join(comps)});")
            self.uses.add(f"fout{rt}")
        if omap2 & 2:
            parts.append(f"gl_FragDepth = uintBitsToFloat({self.r(reg)});")
            reg += 1
        return " ".join(parts) + " return;"

    def assemble(self, body, calls):
        L = ["#version 460", "#extension GL_EXT_nonuniform_qualifier : require",
             "#extension GL_EXT_samplerless_texture_functions : require"]
        if any(u.startswith("img_") for u in self.uses):   # images sans format déclaré (lecture et écriture)
            L.append("#extension GL_EXT_shader_image_load_formatted : require")
        if "int64" in self.uses:
            L.append("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require")
        if "global" in self.uses or self.kind == "vert":
            L.append("#extension GL_EXT_buffer_reference : require")
        if self.kind == "vert":
            L.append("#extension GL_EXT_buffer_reference_uvec2 : require")
        if "tensor" in self.uses:
            L.append("#extension GL_EXT_control_flow_attributes : require")
        if getattr(self, "coopmat", False) and "hmma" in self.uses:
            L += ["#extension GL_KHR_cooperative_matrix : require", "#extension GL_KHR_memory_scope_semantics : require",
                  "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require"]
        if self.uses & {"subgroup", "quad"}:
            L += ["#extension GL_KHR_shader_subgroup_basic : require",
                  "#extension GL_KHR_shader_subgroup_ballot : require",
                  "#extension GL_KHR_shader_subgroup_shuffle : require",
                  "#extension GL_KHR_shader_subgroup_shuffle_relative : require",
                  "#extension GL_KHR_shader_subgroup_vote : require"]
        if "subgroup_arith" in self.uses:
            L.append("#extension GL_KHR_shader_subgroup_arithmetic : require")
        if QUAD_CONTROL and self.kind == "frag" and self.uses & {"subgroup", "quad", "quadp"}:
            L += ["#extension GL_EXT_shader_quad_control : require", "layout(quad_derivatives) in;", "layout(full_quads) in;"]
        if self.banks:
            L.append("#extension GL_EXT_scalar_block_layout : require")
        if self.kind == "comp":
            x, y, z = (max(1, v) for v in self.st.local_size)
            L.append(f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;")
        # Banques de constantes : tampons uniformes Vulkan de 64 Ko (taille d'une banque NVN), tableau serré
        # de mots (disposition scalar). Lues par le cache de constantes du GPU : un déplacement connu devient
        # un opérande direct de l'instruction, comme c[banque][déplacement] sur la console ; en tampons de
        # stockage, chaque lecture était un chargement mémoire.
        for b in sorted(self.banks):
            L.append(f"layout(set = {STAGE_SET[self.kind]}, binding = {b}, scalar) uniform CB{self.sidx}_{b} {{ uint d[16384]; }} cb{self.sidx}_{b};")
        if "cb1" in self.uses:
            w = struct.unpack_from(f"<{len(self.st.cb1) // 4}I", self.st.cb1)
            L.append(f"const uint CB1C[{len(w)}] = uint[]({', '.join(f'0x{x:08x}u' for x in w)});")
        if "global" in self.uses:
            L += self.global_helpers()
        tex_types = {"tex_1d": "1D", "tex_2d": "2D", "tex_3d": "3D", "tex_cube": "Cube", "tex_1darray": "1DArray",
                     "tex_2darray": "2DArray", "tex_cubearray": "CubeArray"}
        TB = {"1D": 0, "2D": 1, "3D": 2, "Cube": 3, "1DArray": 4, "2DArray": 5, "CubeArray": 6}
        used_tex = [tex_types[u] for u in sorted(self.uses) if u in tex_types]
        for ty in used_tex:
            L.append(f"layout(set = {TEX_SET}, binding = {TB[ty]}) uniform texture{ty} T{ty.upper()}[];")
        # images de stockage (jeu 2, liaisons 10 + dim NAK : 10 1D, 12 1DArray, 13 2D, 14 2DArray, 15 3D)
        IB = {"1D": 10, "1DArray": 12, "2D": 13, "2DArray": 14, "3D": 15}
        for ty, bnd in IB.items():
            if "img_" + ty.lower() in self.uses:
                L.append(f"layout(set = {TEX_SET}, binding = {bnd}) uniform image{ty} IMG{ty.upper()}[];")
            if "imgu_" + ty.lower() in self.uses:   # atomiques (SUATOM / SURED) : vue entière r32ui
                L.append(f"layout(set = {TEX_SET}, binding = {bnd}, r32ui) uniform uimage{ty} IMGU{ty.upper()}[];")
        if used_tex:
            L.append(f"layout(set = {TEX_SET}, binding = 8) uniform sampler SMP[];")
        if self.kind == "vert":
            # Position invariante : la passe « standard » teste la profondeur en égalité (EQUAL) contre la
            # pré-passe, dessinée avec un AUTRE shader de sommets ; sur la console les deux calculent la
            # même position au bit près, le pilote Vulkan ne le garantit qu'avec Invariant + NoContraction
            L.append("invariant gl_Position;")
            for k in sorted(self.vin):
                L.append(f"layout(location = {k}) in vec4 vin{k};")
            for k in sorted(self.vout):
                L.append(f"layout(location = {k}) out vec4 vout{k};")
        if self.kind == "frag":
            for k, q in sorted(self.fin.items()):
                L.append(f"layout(location = {k}) {q + ' ' if q else ''}in vec4 fin{k};")
            for rt in range(8):
                if f"fout{rt}" in self.uses:
                    L.append(f"layout(location = {rt}) out vec4 fout{rt};")
        if "shared" in self.uses:
            L.append("shared uint SMEM[12288];")
        if "tensor" in self.uses:   # aides LDSM (SMEM) seulement si le shader a de la mémoire partagée
            aides = [x for x in TENSOR_AIDES if "shared" in self.uses or not any(t in x for t in ("SMEM", "smem16", "ldsm_row"))]
            if getattr(self, "coopmat", False) and "hmma" in self.uses:
                # brouillon des cœurs tenseurs dans SMEM (12 288 mots = 48 Ko, limite de NVIDIA), après la mémoire
                # partagée du programme (mot 0x64 du contrôle), aligné sur 64 octets ; trop grand : pas de variante
                x, y, z = (max(1, v) for v in self.st.local_size)
                nw = (x * y * z + 31) // 32
                used = struct.unpack_from("<I", self.st.ctl.ljust(0x68, bytes(1)), 0x64)[0]
                base = ((used + 63) // 64) * 16
                if base + 256 * nw > 12288 or "shared" not in self.uses:
                    raise Unsupported("cooperative matrix scratch does not fit in shared memory")
                a = aides.index("uvec2 hmma16816f16(uvec4 a, uvec2 b, uvec2 c) {")
                b = aides.index("}", a)
                aides = aides[:a] + aides[b + 1:] + COOPMAT_AIDES(nw, base)
            L += aides
        if getattr(self, "hmma_partage", False):
            L.append("uvec4 HA; uvec2 HB, HC, HD; uint hret;")
        if "hrelu" in self.uses:
            L += HRELU_AIDE
        if "prmt" in self.uses:
            L.append("uint prmt(uint a, uint s, uint b) { uint r = 0u; for (uint k = 0u; k < 4u; k++) {"
                     " uint n = (s >> (4u * k)) & 15u; uint idx = n & 7u; uint w = idx < 4u ? a : b;"
                     " uint by = (w >> (8u * (idx & 3u))) & 255u; if ((n & 8u) != 0u) by = ((by & 128u) != 0u) ? 255u : 0u;"
                     " r |= by << (8u * k); } return r; }")
        if "quadp" in self.uses:
            L += QUAD_AIDES
        if self.kind == "vert":
            L += MESURE_SOMMETS
        L.append("void sar_principal() {" if self.kind == "vert" else "void main() {")
        if "quadp" in self.uses:
            L.append(QUAD_VOIE)
        regs = ", ".join(f"R{k} = 0u" for k in range(self.max_r + 1))
        # sommets : « precise » interdit au pilote de fusionner FMUL + FADD en FMA (seules les FFMA du
        # code d'origine sont fusionnées), pour reproduire exactement les positions
        prec = "precise " if self.kind == "vert" else ""
        L.append(f"  {prec}uint {regs};")
        L.append(f"  {prec}uint " + ", ".join(f"UR{k} = 0u" for k in range(63)) + ";")
        L.append("  bool " + ", ".join(f"P{k} = false" for k in range(7)) + ", " +
                 ", ".join(f"UP{k} = false" for k in range(7)) + ";")
        if "local" in self.uses:
            L.append("  uint LMEM[1024];")
        if calls:
            L.append("  uint cstack[8]; int csp = 0;")
        if self.kind == "vert":
            L.append("  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);")
            for k in sorted(self.vout):
                L.append(f"  vout{k} = vec4(0.0, 0.0, 0.0, 1.0);")
        L.append("  uint pc = 0u;")
        loops = any(i.name in ("brx", "retr") or (i.name == "bra" and i.target is not None and i.target < i.pc)
                    for i in self.insns)
        if loops:
            # Garde-fou : au-delà d'un million de transitions de bloc, l'invocation s'arrête. Le shader
            # du matériau principal de la cinématique faisait réinitialiser le GPU (TDR, périphérique perdu)
            # sans boucle infinie mesurable ; avec ce compteur, le pilote le compile autrement et il
            # s'exécute normalement (une invocation saine reste très loin de cette borne).
            L.append("  uint guard = 0u;")
        if os.environ.get("RECOMP_SHADER_SWITCH"):   # forme d'origine, pour comparer (voir en_chaine)
            L.append("  while (true) {")
            if loops:
                L.append("    if (++guard > 1000000u) return;")
            L.append("    switch (pc) {")
            L += body
            L.append("    default: return;")
            L.append("    }")
            L.append("    return;")
            L.append("  }")
        else:
            L += en_chaine(body, loops)
        L.append("}")
        if self.kind == "vert":   # le corps traduit, puis la mesure de distance (sans effet hors des dessins mesurés)
            L += ["void main() {", "  sar_principal();", "  sar_mesure();", "}"]
        return "\n".join(L) + "\n"


def en_chaine(body, boucle):
    """Corps de main() à partir des blocs `case N:` de l'émetteur, sans boucle de répartition.

    Jusqu'à la 1.0.2 : `while (true) { switch (pc) { case N: … pc = M; continue; … default: return; } return; }`.
    Les compilateurs des pilotes devaient retrouver le vrai flot de contrôle derrière l'aiguillage ; celui de NVIDIA y
    arrive, celui d'AMD non (fondé sur LLVM, qui ne relie pas un saut à sa cible à travers l'en-tête d'une boucle) :
    chaque pixel exécutait la boucle et l'aiguillage. Mesuré sur une carte AMD intégrée en 1080p : 0,1 image par
    seconde dans les scènes 3D, image fausse ; compilation des shaders en plus de 4 minutes (issues n° 6, 7 et 11).

    Maintenant chaque bloc est `if (pc == N) do { … pc = M; break; … } while (false);`, dans l'ordre des adresses.
    Sans saut arrière ni saut calculé (`boucle` faux, 1 953 shaders sur 2 388), tous les sauts vont vers un bloc
    écrit plus loin : une seule passe suffit, il ne reste ni boucle ni aiguillage, et le compilateur relie chaque saut
    à sa cible. Sinon (435 shaders) la chaîne est répétée : un saut arrière prend effet à la passe suivante, et une
    passe où aucun bloc ne s'exécute termine le shader (comme `default: return`). Même carte, mêmes scènes : 2,5 à 13
    images par seconde, compilation en 53 s."""
    blocs, cur = [], None
    for l in body:
        m = re.match(r"    case (\d+)u:\s*(.*)$", l)
        if m:
            cur = [int(m.group(1)), []]
            blocs.append(cur)
            if m.group(2).strip():
                cur[1].append("      " + m.group(2))
        else:
            cur[1].append(l)
    etiquettes = [pc for pc, _ in blocs]
    garde = boucle   # (le compteur `guard` n'est déclaré que pour les shaders qui bouclent)
    if etiquettes != sorted(etiquettes) or len(set(etiquettes)) != len(etiquettes):
        boucle = True
    for pc, corps in blocs:
        for l in corps:
            code = l.split("  //")[0]
            if re.search(r"\bbreak;", code):
                raise Unsupported("break in a block")
            for cible in re.findall(r"\bpc = ([^;]+);", code):   # saut constant vers l'arrière, ou saut calculé
                cible = cible.strip()
                if not re.fullmatch(r"\d+u", cible) or int(cible[:-1]) <= pc:
                    boucle = True
    out, ind = [], "  "
    if boucle:
        out.append("  while (true) {")
        if garde:
            out.append("    if (++guard > 1000000u) return;")
        out.append("    bool passe = false;")
        ind = "    "
    for k, (pc, corps) in enumerate(blocs):
        out.append(f"{ind}if (pc == {pc}u) do {{")
        if boucle:
            out.append(f"{ind}  passe = true;")
        dernier = ""
        for l in corps:
            out.append(re.sub(r"\bcontinue;", "break;", l))
            code = l.split("  //")[0].strip()
            if code:
                dernier = code
        if not re.fullmatch(r"(pc = [^;]*; )?(continue|return);", dernier):   # le bloc tombait dans le suivant
            out.append(f"{ind}  pc = {blocs[k + 1][0] if k + 1 < len(blocs) else 0xffffffff}u;")
        out.append(f"{ind}}} while (false);")
    if boucle:
        out.append("    if (!passe) return;")
        out.append("  }")
    return out


def translate(stage, coopmat=False):
    """coopmat : variante où HMMA passe par les cœurs tenseurs du GPU (COOPMAT_AIDES) ; seulement si le shader en a."""
    e = Emitter(stage)
    e.coopmat = coopmat
    src = e.translate()
    return src, e


def metadata(e):
    """Description lue par le runtime (vk_render.cpp) : une information par ligne."""
    tex = sorted(u[4:] for u in e.uses if u.startswith("tex_"))
    img = sorted(f"{b}:{o:#x}:{t}" for b, o, t in e.img_handles)
    texh = sorted(f"{b}:{o:#x}" for b, o in e.tex_handles)
    fout = sorted(int(u[4:]) for u in e.uses if u.startswith("fout"))
    lines = [f"stage {e.kind}",
             f"banks {' '.join(str(b) for b in sorted(e.banks))}",
             f"vin {' '.join(str(k) for k in sorted(e.vin))}",
             f"vout {' '.join(str(k) for k in sorted(e.vout))}",
             f"fin {' '.join(str(k) for k in sorted(e.fin))}",
             f"fout {' '.join(str(k) for k in fout)}",
             f"tex {' '.join(tex)}",
             f"img {' '.join(img)}",
             f"texh {' '.join(texh)}"]
    if e.kind == "comp":
        lines.append("local " + " ".join(str(max(1, v)) for v in e.st.local_size))
        # banque 0 du calcul (établi sur 10 étages, taille totale = mot 0xb8 du contrôle) : en-tête 0x10,
        # 16 o par tampon uniforme, 8 o par texture puis 8 o par image ; nombres aux octets 0xb1, 0xb3, 0xb6
        # + mots de données en ligne (octet 0xa9, placées en 0x10), tampons de stockage (0xb2), textures et
        # échantillonneurs séparés (0xb4, 0xb5, entre les textures et les images ; vérifié sur la lumière du
        # brouillard : 0x10 + 13 × 16 + 4 × 8 + 32 × 8 + 13 × 8 + 5 × 8 = 0x290, taille de l'octet 0xba)
        ctl = getattr(e.st, "ctl", b"")
        if len(ctl) >= 0xb8:
            lines.append(f"c0 {ctl[0xb1]} {ctl[0xb3]} {ctl[0xb6]} {ctl[0xa9]} {ctl[0xb2]} {ctl[0xb4]} {ctl[0xb5]}")
    if "global" in e.uses:   # calculs, et le seul étage graphique concerné (color_plot_draw.frag, f979ce45bdf2392a)
        lines.append("glob 1")
    return "\n".join(lines) + "\n"
