/**
 * =============================================================================
 * qlearning.js — Q-Learning algoritam za PONG AI Training Lab
 * =============================================================================
 *
 * Implementacija tabelarnog Q-Learning algoritma (Watkins & Dayan, 1992).
 * Ovaj fajl je namjerno izolovan od ostatka aplikacije — sve što je ovdje
 * je čista AI logika, bez UI-a, bez DOM-a, bez Pong fizike.
 *
 * Koristi se iz index.html i game loop-a (simGame funkcija).
 * =============================================================================
 */


// =============================================================================
// KONSTANTE — dimenzije prostora stanja i akcija
// =============================================================================

// Dimenzije canvas platna (mora biti konzistentno s game loop-om)
const CW = 800, CH = 500;

// Fizičke dimenzije paddle-a i loptice
const PADDLE_W = 12, PADDLE_H = 80;
const BALL_SIZE = 10;
const PADDLE_SPEED = 5;

/**
 * STATE SPACE DIZAJN — diskretizacija kontinualnog prostora
 *
 * Agent ne vidi sirove koordinate (floating point) nego "binove" —
 * diskretne kategorije. To je neophodno jer tabelarni Q-Learning
 * zahtijeva konačan, prebrojiv broj stanja.
 *
 * Analogija: umjesto da pamtiš tačnu temperaturu (36.743°C), pamtiš
 * kategoriju (toplo / vruće / hladno). Gubi se preciznost, ali
 * agent može generalizovati između sličnih situacija.
 *
 * 6 dimenzija × njihovi binovi:
 *
 *   Dimenzija         | Binovi | Šta opisuje
 *   ──────────────────┼────────┼─────────────────────────────────────
 *   Pozicija loptice X|  10    | Horizontalna zona (0=lijevo, 9=desno)
 *   Pozicija loptice Y|  10    | Vertikalna zona (0=gore, 9=dolje)
 *   Smjer loptice dX  |   3    | Ide lijevo (0) / neutralno (1) / desno (2)
 *   Brzina loptice dY |   5    | Vertikalna brzina u 5 razreda
 *   Pozicija paddlea  |   8    | Vlastiti paddle, 8 vertikalnih zona
 *   Pozicija protivn. |   6    | Protivnički paddle, 6 zona
 *   ──────────────────┼────────┼─────────────────────────────────────
 *   UKUPNO            |10×10×3×5×8×6 = 72.000 stanja
 */
const BALL_X_BINS  = 10;
const BALL_Y_BINS  = 10;
const BALL_DX_BINS = 3;  // 0=lijevo, 1=neutralno, 2=desno
const BALL_DY_BINS = 5;  // vertikalna brzina u 5 razreda
const PADDLE_Y_BINS = 8; // vlastiti paddle
const OPP_Y_BINS   = 6;  // protivnički paddle

/**
 * Ukupan broj stanja = produkt svih binova.
 * Svako stanje ima ACTIONS moguće akcije → Q-table ima STATE_SIZE × ACTIONS ćelija.
 */
const STATE_SIZE = BALL_X_BINS * BALL_Y_BINS * BALL_DX_BINS * BALL_DY_BINS * PADDLE_Y_BINS * OPP_Y_BINS;

/**
 * Akcije agenta:
 *   0 = pomjeri paddle gore
 *   1 = ostani na mjestu
 *   2 = pomjeri paddle dolje
 */
const ACTIONS = 3;


// =============================================================================
// QLearningAgent — klasa koja enkapsulira cijeli Q-Learning algoritam
// =============================================================================

class QLearningAgent {

  /**
   * Konstruktor — inicijalizuje agenta s parametrima učenja.
   *
   * @param {Object} params - Hyperparametri i metadata agenta
   * @param {string} params.name         - Ime agenta (za UI i export)
   * @param {number} params.lr           - Learning rate α (0.001–1.0)
   * @param {number} params.discount     - Discount factor γ (0.5–0.999)
   * @param {number} params.epsilon      - Početna vjerovatnoća eksploracije (0–1)
   * @param {number} params.epsilonDecay - Faktor smanjenja epsilon-a po epizodi
   * @param {number} params.trainSpeed   - Brzina loptice tokom treninga
   */
  constructor(params) {
    this.name         = params.name;
    this.lr           = params.lr           || 0.1;
    this.discount     = params.discount     || 0.95;
    this.epsilon      = params.epsilon      || 0.1;
    this.epsilonDecay = params.epsilonDecay || 0.9995;
    this.epsilonMin   = 0.01;  // Epsilon nikad ne pada ispod ovoga — uvijek neka eksploracija
    this.trainSpeed   = params.trainSpeed   || 5;

    // Statistike
    this.totalEpisodes = 0;
    this.wins   = 0;
    this.losses = 0;
    this.rating = 1000;  // Početni Elo-like rating
    this.trainLog = [];
    this.trained  = false;

    /**
     * Q-TABLE — srž algoritma.
     *
     * Implementirana kao flat (jednodimenzionalni) Float32Array umjesto
     * 2D matrice, zbog memorijske efikasnosti i brzine pristupa.
     *
     * Veličina: STATE_SIZE × ACTIONS = 72.000 × 3 = 216.000 float32 vrijednosti
     * Memorija: 216.000 × 4 bajta = ~864 KB po agentu
     *
     * Indeksiranje: Q[stanje * ACTIONS + akcija]
     * Primjer: Q vrijednost za stanje 5000, akcija 2 (dolje) → Q[5000*3 + 2] = Q[15002]
     *
     * Početna vrijednost svih ćelija je 0 — optimistični pristup bi koristio
     * pozitivne vrijednosti da forsira eksploraciju, ali 0 radi dobro uz epsilon.
     */
    this.Q = new Float32Array(STATE_SIZE * ACTIONS).fill(0);
  }


  /**
   * encodeState — pretvara kontinualne vrijednosti igre u diskretni indeks stanja.
   *
   * Ovo je "most" između fizike igre i Q-tabele. Prima 6 floating-point
   * vrijednosti i vraća jedan integer u rasponu [0, STATE_SIZE-1].
   *
   * Tehnika kodiranja: mixed-radix encoding (kao što su sat/minute/sekunde).
   * Svaka dimenzija ima svoj "radix" (broj binova), i kombiniraju se
   * množeći i sabérajući — kao cifre broja u različitim bazama.
   *
   * @param {number} bx  - X pozicija loptice (0–CW)
   * @param {number} by  - Y pozicija loptice (0–CH)
   * @param {number} bdx - Horizontalna brzina loptice (negativna=lijevo)
   * @param {number} bdy - Vertikalna brzina loptice
   * @param {number} py  - Y pozicija vlastitog paddlea (0–CH)
   * @param {number} opy - Y pozicija protivničkog paddlea (0–CH)
   * @returns {number} Indeks stanja u rasponu [0, STATE_SIZE-1]
   */
  encodeState(bx, by, bdx, bdy, py, opy) {
    // Svaka dimenzija se normalizuje (dijeli s max) pa množi s brojem binova
    // Math.floor → cijeli broj, Math.min/max → clamp da ne izađemo iz opsega

    // X pozicija loptice: 0 (lijevi rub) do BALL_X_BINS-1 (desni rub)
    const bxi = Math.min(BALL_X_BINS-1,  Math.max(0, Math.floor(bx/CW * BALL_X_BINS)));

    // Y pozicija loptice
    const byi = Math.min(BALL_Y_BINS-1,  Math.max(0, Math.floor(by/CH * BALL_Y_BINS)));

    // Smjer loptice — NIJE linearan bin nego threshold klasifikacija:
    //   bdx < -1 → ide lijevo (0)
    //   -1 ≤ bdx ≤ 1 → neutralno (1)  [rijetko u praksi]
    //   bdx > 1  → ide desno (2)
    const bdxi = bdx < -1 ? 0 : bdx > 1 ? 2 : 1;

    // Vertikalna brzina — koristi apsolutnu vrijednost (smjer nije bitan,
    // samo koliko brzo se loptica kreće gore/dolje)
    const bspeed = Math.abs(bdy);
    const bdyi = Math.min(BALL_DY_BINS-1, Math.max(0, Math.floor(bspeed / 3 * BALL_DY_BINS)));

    // Pozicija vlastitog paddlea
    const pyi  = Math.min(PADDLE_Y_BINS-1, Math.max(0, Math.floor(py/CH * PADDLE_Y_BINS)));

    // Pozicija protivničkog paddlea
    const opyi = Math.min(OPP_Y_BINS-1,   Math.max(0, Math.floor(opy/CH * OPP_Y_BINS)));

    // Mixed-radix encoding — kombinuje sve dimenzije u jedan broj
    // Redosljed: bxi → byi → bdxi → bdyi → pyi → opyi
    // Svaka dimenzija "pomiče" prethodne za njen broj binova
    return ((bxi * BALL_Y_BINS + byi) * BALL_DX_BINS + bdxi) * BALL_DY_BINS * PADDLE_Y_BINS * OPP_Y_BINS +
            bdyi * PADDLE_Y_BINS * OPP_Y_BINS + pyi * OPP_Y_BINS + opyi;
  }


  /**
   * getAction — odlučuje koja akcija se poduzima u datom stanju.
   *
   * Implementira epsilon-greedy politiku — balans između:
   *   - EKSPLORACIJE: nasumična akcija (otkrivanje novih mogućnosti)
   *   - EKSPLOATACIJE: best poznata akcija iz Q-tabele (korišćenje naučenog)
   *
   * Na početku treninga epsilon je visok (npr. 1.0) → skoro sve je eksploracija.
   * Tokom treninga epsilon opada → agent sve više koristi naučeno znanje.
   *
   * Zašto je eksploracija neophodna? Bez nje agent bi zauvijek koristio
   * prvu akciju koja je dala ikakvu nagradu, nikad ne otkrivajući bolja rješenja.
   *
   * @param {number}  state   - Indeks stanja (iz encodeState)
   * @param {boolean} explore - False = uvijek eksploatacija (za live igru bez treninga)
   * @returns {number} Akcija: 0=gore, 1=stoj, 2=dolje
   */
  getAction(state, explore=true) {
    // Eksploracija — nasumična akcija s vjerovatnoćom epsilon
    if (explore && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * ACTIONS);
    }

    // Eksploatacija — traži akciju s najvišom Q vrijednošću za ovo stanje
    // Q vrijednosti su na lokacijama: state*ACTIONS, state*ACTIONS+1, state*ACTIONS+2
    let best = 0, bestQ = this.Q[state * ACTIONS];
    for (let a = 1; a < ACTIONS; a++) {
      const q = this.Q[state * ACTIONS + a];
      if (q > bestQ) { bestQ = q; best = a; }
    }
    return best;
  }


  /**
   * update — Bellman jednadžba, jedino mjesto gdje agent uči.
   *
   * Ovo je srž Q-Learning algoritma. Poziva se nakon svakog koraka igre
   * s četiri podatka: stanje prije, akcija, nagrada, stanje poslije.
   *
   * Bellman jednadžba:
   *   Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
   *
   * Intuicija: "Koliko vrijedi biti u stanju s i poduzeti akciju a?"
   *   - r                    = trenutna nagrada (šta si odmah dobio)
   *   - γ · max Q(s',a')     = diskontovana buduća nagrada (šta možeš još dobiti)
   *   - r + γ · max Q(s',a') = "target" — šta Q(s,a) TREBA biti
   *   - target - Q(s,a)      = TD error (Temporal Difference) — koliko griješimo
   *   - α · TD_error         = korekcija — pomjeramo Q prema targetu za mali korak
   *
   * Primjer iz Ponga:
   *   Agent je u stanju s (loptica dolazi desno, paddle u sredini).
   *   Pomjeri se gore (a=0). Loptica ga promašuje, gubi bod (r=-1).
   *   Sljedeće stanje s' — Q(s', best_a') je malo pozitivno.
   *   TD error je negativan → Q(s, gore) se smanjuje.
   *   Sljedeći put agent u toj situaciji neće ići gore.
   *
   * @param {number} s  - Stanje PRIJE akcije
   * @param {number} a  - Poduzeta akcija (0/1/2)
   * @param {number} r  - Nagrada: +1=pogodak paddlea, -1=propuštena loptica
   * @param {number} s2 - Stanje POSLIJE akcije (s' u Bellmanovoj jednadžbi)
   */
  update(s, a, r, s2) {
    // Indeks u flat arrayu za Q(s,a)
    const idx = s * ACTIONS + a;

    // Traži max Q(s', a') — best moguća buduća nagrada iz sljedećeg stanja
    // Ovo je "optimistična" procjena budućnosti (pretpostavljamo optimalni play)
    let maxQ2 = this.Q[s2 * ACTIONS];
    for (let a2 = 1; a2 < ACTIONS; a2++) {
      if (this.Q[s2 * ACTIONS + a2] > maxQ2) maxQ2 = this.Q[s2 * ACTIONS + a2];
    }

    // Bellman update — pomjeramo Q(s,a) prema targetu za korak α
    // this.lr   = α (learning rate)
    // this.discount = γ (discount factor)
    this.Q[idx] += this.lr * (r + this.discount * maxQ2 - this.Q[idx]);
  }


  /**
   * decayEpsilon — smanjuje epsilon nakon svake epizode.
   *
   * Multiplikativni decay: epsilon = epsilon × epsilonDecay
   * Primjer s epsilonDecay=0.9995 i 10.000 epizoda:
   *   Start:   epsilon = 1.0
   *   1.000 ep: epsilon ≈ 0.607
   *   5.000 ep: epsilon ≈ 0.082
   *   10.000 ep: epsilon ≈ 0.007 → clampuje na epsilonMin=0.01
   *
   * Rezultat: agent prelazi iz "skoro sve eksploriraj" u "skoro sve eksploatiraj"
   * tokom treninga, ali uvijek zadržava minimalnu eksploraciju (epsilonMin).
   */
  decayEpsilon() {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }


  /**
   * ratingFactor — penalizuje agente koji su puno trenirani.
   *
   * Sprečava da "brute-force" trening dominira ljestvicu.
   * Agent koji pobijedi s 100 epizoda zaslužuje više poena nego
   * agent koji pobijedi s 100.000 epizoda.
   *
   * Formula: max(0.2, 1.5 - log10(epizode) × 0.15)
   *   100 ep    → faktor ≈ 1.20
   *   1.000 ep  → faktor ≈ 1.05
   *   10.000 ep → faktor ≈ 0.90
   *   100.000ep → faktor ≈ 0.75
   *
   * @returns {number} Množilac poena u rasponu [0.2, ~1.5]
   */
  ratingFactor() {
    if (this.totalEpisodes === 0) return 1.0;
    const base = Math.log10(Math.max(10, this.totalEpisodes));
    return Math.max(0.2, 1.5 - base * 0.15);
  }


  /**
   * toJSON — serijalizuje agenta za export i localStorage.
   *
   * Float32Array mora biti konvertovan u obični Array jer JSON.stringify
   * ne može direktno serijalizovati typed arrays.
   * Veličina eksportovanog JSON-a: ~864KB (Q-table) + metadata.
   *
   * @returns {Object} Plain object spreman za JSON.stringify
   */
  toJSON() {
    return {
      name: this.name, lr: this.lr, discount: this.discount,
      epsilon: this.epsilon, epsilonDecay: this.epsilonDecay,
      trainSpeed: this.trainSpeed, totalEpisodes: this.totalEpisodes,
      wins: this.wins, losses: this.losses, rating: this.rating,
      trainLog: this.trainLog, trained: this.trained,
      Q: Array.from(this.Q)  // Float32Array → obični Array za JSON
    };
  }


  /**
   * fromJSON — deserijalizuje agenta iz importovanog JSON-a.
   *
   * Static metoda (factory) — kreira novi QLearningAgent iz plain objekta.
   * Q-table se vraća u Float32Array za efikasnost.
   *
   * @param {Object} d - Plain object (iz JSON.parse)
   * @returns {QLearningAgent} Potpuno rekonstruisan agent
   */
  static fromJSON(d) {
    const a = new QLearningAgent(d);
    a.totalEpisodes = d.totalEpisodes;
    a.wins   = d.wins;
    a.losses = d.losses;
    a.rating = d.rating;
    a.trainLog = d.trainLog || [];
    a.trained  = d.trained;
    a.Q = new Float32Array(d.Q);  // obični Array → Float32Array
    return a;
  }
}
