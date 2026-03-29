/**
 * =============================================================================
 * qlearning.js — Q-Learning algoritam za PONG AI Training Lab
 * =============================================================================
 *
 * Implementacija tabelarnog Q-Learning algoritma (Watkins & Dayan, 1992)
 * s DQN stabilizacijskim tehnikama (DeepMind, 2013/2015):
 *   - Experience Replay Buffer (Lin, 1992)
 *   - Target Network (Mnih et al., 2015)
 *
 * Ovaj fajl je namjerno izolovan od ostatka aplikacije — sve što je ovdje
 * je čista AI logika, bez UI-a, bez DOM-a, bez Pong fizike.
 * =============================================================================
 */


// =============================================================================
// KONSTANTE — dimenzije prostora stanja i akcija
// =============================================================================

const CW = 800, CH = 500;
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
 *   UKUPNO            | 10×10×3×5×8×6 = 72.000 stanja
 */
const BALL_X_BINS  = 10;
const BALL_Y_BINS  = 10;
const BALL_DX_BINS = 3;
const BALL_DY_BINS = 5;
const PADDLE_Y_BINS = 8;
const OPP_Y_BINS   = 6;

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
   * @param {string} params.name              - Ime agenta
   * @param {number} params.lr                - Learning rate α (0.001–1.0)
   * @param {number} params.discount          - Discount factor γ (0.5–0.999)
   * @param {number} params.epsilon           - Početna vjerovatnoća eksploracije
   * @param {number} params.epsilonDecay      - Faktor smanjenja epsilon-a po epizodi
   * @param {number} params.trainSpeed        - Brzina loptice tokom treninga
   * @param {number} params.bufferSize        - Kapacitet replay buffera (default: 5000)
   * @param {number} params.batchSize         - Veličina mini-batcha (default: 32)
   * @param {number} params.targetSyncInterval- Svakih N epizoda sinhronizuj target mrežu (default: 300)
   */
  constructor(params) {
    this.name         = params.name;
    this.lr           = params.lr           || 0.1;
    this.discount     = params.discount     || 0.95;
    this.epsilon      = params.epsilon      || 1.0;
    this.epsilonDecay = params.epsilonDecay || 0.9995;
    this.epsilonMin   = 0.01;
    this.trainSpeed   = params.trainSpeed   || 5;

    // Experience Replay parametri
    this.bufferSize         = params.bufferSize        || 5000;
    this.batchSize          = params.batchSize         || 32;
    this.targetSyncInterval = params.targetSyncInterval || 300;

    // Statistike
    this.totalEpisodes = 0;
    this.wins   = 0;
    this.losses = 0;
    this.rating = 1000;
    this.trainLog = [];
    this.trained  = false;

    // -------------------------------------------------------------------------
    // ONLINE Q-TABLE — ažurira se svaki korak učenja
    //
    // Ovo je "radna" Q-tabela — ona koja prima sve update-ove i čija se
    // vrijednost mijenja u svakom koraku treninga.
    // -------------------------------------------------------------------------
    this.Q = new Float32Array(STATE_SIZE * ACTIONS).fill(0);

    // -------------------------------------------------------------------------
    // TARGET Q-TABLE — zamrznuta kopija online Q-tabele
    //
    // Ideja DeepMind-a (2015): umjesto da Bellmanova jednadžba koristi
    // istu tabelu za računanje i za update, koristimo DVIJE tabele:
    //   - Online Q: prima sve update-ove, uči se svaki korak
    //   - Target Q: zamrznuta kopija, koristi se samo za maxQ(s',a')
    //
    // Zašto je ovo važno?
    // Zamislimo da učimo procjenu "koliko vrijedi biti u nekom stanju".
    // Ako istovremeno mijenjamo i procjenu i cilj prema kojoj učimo,
    // učenje postaje nestabilno — kao pokušaj gađanja mete koja se kreće.
    // Target Q drži "cilj" stabilnim dok online Q konvergira prema njemu.
    //
    // Sinhronizacija: svakih targetSyncInterval epizoda, online Q se
    // kopira u target Q (hard update). U DQN-u postoji i "soft update"
    // varijanta (Polyak averaging), ali hard update je jednostavniji
    // i dovoljno dobar za tabelarni slučaj.
    // -------------------------------------------------------------------------
    this.targetQ = new Float32Array(STATE_SIZE * ACTIONS).fill(0);
    this.episodesSinceSync = 0;

    // -------------------------------------------------------------------------
    // EXPERIENCE REPLAY BUFFER — circular buffer iskustava
    //
    // Ideja (Lin, 1992; DeepMind, 2015): svaki korak igre generira jedno
    // iskustvo (s, a, r, s') — "u stanju s, poduzeo sam akciju a,
    // dobio sam nagradu r, i završio u stanju s'".
    //
    // Bez replay buffera: agent uči direktno od svake interakcije u nizu.
    // Problem: uzastopna iskustva su visoko korelirana (loptica se kreće
    // glatko, stanja su slična). Učenje iz koreliranih podataka je
    // nestabilno i vodi u "zaboravljanje" ranijeg znanja.
    //
    // S replay bufferom:
    //   1. Svako iskustvo se sprema u buffer (kapaciteta bufferSize)
    //   2. Stara iskustva se automatski brišu kad buffer popuni (circular)
    //   3. Na kraju svakog koraka, nasumično se uzorkuje mini-batch iz buffera
    //   4. Agent uči iz tog nasumičnog batcha — korelacija je slomljena
    //
    // Efekti:
    //   - Svako iskustvo se "reciklira" — koristi se više puta za učenje
    //   - Nasumično uzorkovanje razbija vremensku korelaciju
    //   - Trening je stabilniji, agent uči efikasnije
    //
    // Implementacija: flat Int32Array s 4 vrijednosti po iskustvu (s, a, r, s2)
    // r se čuva kao integer (-1, 0, +1) jer su nagrade u Pongu uvijek cijele.
    // -------------------------------------------------------------------------
    this.buffer    = new Int32Array(this.bufferSize * 4); // [s, a, r, s2] × bufferSize
    this.bufferLen = 0;   // koliko iskustava je trenutno u bufferu
    this.bufferPtr = 0;   // index sljedećeg pisanja (circular)
  }


  /**
   * encodeState — pretvara kontinualne vrijednosti igre u diskretni indeks stanja.
   *
   * Tehnika: mixed-radix encoding (analogija: sat/minute/sekunde).
   *
   * @param {number} bx  - X pozicija loptice (0–CW)
   * @param {number} by  - Y pozicija loptice (0–CH)
   * @param {number} bdx - Horizontalna brzina loptice
   * @param {number} bdy - Vertikalna brzina loptice
   * @param {number} py  - Y pozicija vlastitog paddlea
   * @param {number} opy - Y pozicija protivničkog paddlea
   * @returns {number} Indeks stanja [0, STATE_SIZE-1]
   */
  encodeState(bx, by, bdx, bdy, py, opy) {
    const bxi  = Math.min(BALL_X_BINS-1,   Math.max(0, Math.floor(bx/CW * BALL_X_BINS)));
    const byi  = Math.min(BALL_Y_BINS-1,   Math.max(0, Math.floor(by/CH * BALL_Y_BINS)));
    const bdxi = bdx < -1 ? 0 : bdx > 1 ? 2 : 1;
    const bspeed = Math.abs(bdy);
    const bdyi = Math.min(BALL_DY_BINS-1,  Math.max(0, Math.floor(bspeed / 3 * BALL_DY_BINS)));
    const pyi  = Math.min(PADDLE_Y_BINS-1, Math.max(0, Math.floor(py/CH * PADDLE_Y_BINS)));
    const opyi = Math.min(OPP_Y_BINS-1,    Math.max(0, Math.floor(opy/CH * OPP_Y_BINS)));
    return ((bxi * BALL_Y_BINS + byi) * BALL_DX_BINS + bdxi) * BALL_DY_BINS * PADDLE_Y_BINS * OPP_Y_BINS +
            bdyi * PADDLE_Y_BINS * OPP_Y_BINS + pyi * OPP_Y_BINS + opyi;
  }


  /**
   * getAction — epsilon-greedy politika odlučivanja.
   *
   * @param {number}  state   - Indeks stanja
   * @param {boolean} explore - False = uvijek eksploatacija (live igra)
   * @returns {number} Akcija: 0=gore, 1=stoj, 2=dolje
   */
  getAction(state, explore=true) {
    if (explore && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * ACTIONS);
    }
    let best = 0, bestQ = this.Q[state * ACTIONS];
    for (let a = 1; a < ACTIONS; a++) {
      const q = this.Q[state * ACTIONS + a];
      if (q > bestQ) { bestQ = q; best = a; }
    }
    return best;
  }


  /**
   * pushReplay — sprema jedno iskustvo u replay buffer.
   *
   * Circular buffer: kad se popuni, stara iskustva se prepisuju.
   * Svako iskustvo = (s, a, r, s2) = 4 integera.
   *
   * @param {number} s  - Stanje prije akcije
   * @param {number} a  - Poduzeta akcija (0/1/2)
   * @param {number} r  - Nagrada (-1, 0, +1)
   * @param {number} s2 - Stanje nakon akcije
   */
  pushReplay(s, a, r, s2) {
    const i = this.bufferPtr * 4;
    this.buffer[i]   = s;
    this.buffer[i+1] = a;
    this.buffer[i+2] = r;  // -1/0/+1 — sigurno u Int32Array
    this.buffer[i+3] = s2;
    this.bufferPtr = (this.bufferPtr + 1) % this.bufferSize; // circular wrap
    if (this.bufferLen < this.bufferSize) this.bufferLen++;
  }


  /**
   * learnFromReplay — uči iz nasumičnog mini-batcha iz replay buffera.
   *
   * Koristi TARGET Q-tabelu za računanje maxQ(s',a') —
   * to je ključna razlika od klasičnog Q-Learning update-a.
   *
   * Bellman jednadžba s target networkom:
   *   target = r + γ · max_a' targetQ(s', a')
   *   Q(s,a) ← Q(s,a) + α · [target − Q(s,a)]
   *
   * Nasumično uzorkovanje: Fisher-Yates shuffle prvih batchSize indeksa,
   * ili jednostavno slučajni indeks po pozivu — biramo drugi pristup
   * za brzinu (nema alokacije).
   *
   * @param {number} [batchSize] - Broj iskustava za učenje (default: this.batchSize)
   */
  learnFromReplay(batchSize) {
    const bs = batchSize || this.batchSize;
    if (this.bufferLen < bs) return; // čekamo dok se buffer dovoljno napuni

    for (let b = 0; b < bs; b++) {
      // Nasumični indeks iz popunjenog dijela buffera
      const idx = Math.floor(Math.random() * this.bufferLen);
      const i = idx * 4;
      const s  = this.buffer[i];
      const a  = this.buffer[i+1];
      const r  = this.buffer[i+2];
      const s2 = this.buffer[i+3];

      // maxQ(s',a') iz TARGET Q-tabele (zamrznuta referenca)
      let maxTargetQ = this.targetQ[s2 * ACTIONS];
      for (let a2 = 1; a2 < ACTIONS; a2++) {
        const tq = this.targetQ[s2 * ACTIONS + a2];
        if (tq > maxTargetQ) maxTargetQ = tq;
      }

      // Bellman update na ONLINE Q-tabeli
      const qIdx = s * ACTIONS + a;
      this.Q[qIdx] += this.lr * (r + this.discount * maxTargetQ - this.Q[qIdx]);
    }
  }


  /**
   * syncTargetNetwork — kopira online Q u target Q (hard update).
   *
   * Poziva se svakih targetSyncInterval epizoda.
   * Float32Array.set() je O(n) memcpy — brzo.
   */
  syncTargetNetwork() {
    this.targetQ.set(this.Q);
    this.episodesSinceSync = 0;
  }


  /**
   * update — direktni Bellman update (legacy, koristi se za backward compatibility).
   *
   * Novi kod koristi pushReplay + learnFromReplay.
   * Ova metoda ostaje za agente koji su kreirani bez replay buffera
   * (importirani stari JSON-ovi bez buffer polja).
   *
   * @param {number} s  - Stanje prije
   * @param {number} a  - Akcija
   * @param {number} r  - Nagrada
   * @param {number} s2 - Stanje poslije
   */
  update(s, a, r, s2) {
    const idx = s * ACTIONS + a;
    let maxQ2 = this.Q[s2 * ACTIONS];
    for (let a2 = 1; a2 < ACTIONS; a2++) {
      if (this.Q[s2 * ACTIONS + a2] > maxQ2) maxQ2 = this.Q[s2 * ACTIONS + a2];
    }
    this.Q[idx] += this.lr * (r + this.discount * maxQ2 - this.Q[idx]);
  }


  /**
   * decayEpsilon — smanjuje epsilon multiplikativno.
   *
   * epsilon = max(epsilonMin, epsilon × epsilonDecay)
   * Poziva se jednom po epizodi (ne po koraku).
   */
  decayEpsilon() {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }


  /**
   * ratingFactor — penalizuje agente s mnogo treninga.
   *
   * Formula: max(0.2, 1.5 - log10(epizode) × 0.15)
   */
  ratingFactor() {
    if (this.totalEpisodes === 0) return 1.0;
    const base = Math.log10(Math.max(10, this.totalEpisodes));
    return Math.max(0.2, 1.5 - base * 0.15);
  }


  /**
   * toJSON — serijalizuje agenta za export i localStorage.
   *
   * Replay buffer se NE exportuje (privremeni trening data, ~80KB).
   * Target Q se NE exportuje (rekonstruiše se pri prvom sync-u).
   * Export ostaje iste veličine kao prije (~864KB Q-table + metadata).
   */
  toJSON() {
    // Sparse encoding Q-table: cuvamo samo nenulte vrijednosti kao [index, value] parove.
    // Stedi ~70-90% prostora jer vecina Q vrijednosti ostaje 0.
    // Format: {i: [idx0, idx1, ...], v: [val0, val1, ...]}
    const qi = [], qv = [];
    for (let i = 0; i < this.Q.length; i++) {
      if (this.Q[i] !== 0) {
        qi.push(i);
        qv.push(Math.round(this.Q[i] * 10000) / 10000); // 4 decimale dovoljno
      }
    }
    return {
      name: this.name, lr: this.lr, discount: this.discount,
      epsilon: this.epsilon, epsilonDecay: this.epsilonDecay,
      trainSpeed: this.trainSpeed, totalEpisodes: this.totalEpisodes,
      wins: this.wins, losses: this.losses, rating: this.rating,
      trainLog: this.trainLog, trained: this.trained,
      bufferSize: this.bufferSize, batchSize: this.batchSize,
      targetSyncInterval: this.targetSyncInterval,
      Qs: {i: qi, v: qv}  // sparse format
    };
  }


  /**
   * fromJSON — deserijalizuje agenta iz JSON-a (backward compatible).
   *
   * Stari agenti bez bufferSize/batchSize/targetSyncInterval dobivaju
   * default vrijednosti — rade s novim algoritmom bez problema.
   */
  // toJSONExport — dense format za download/analizu (backward compatible)
  toJSONExport() {
    const base = this.toJSON();
    delete base.Qs;
    base.Q = Array.from(this.Q);
    return base;
  }

  static fromJSON(d) {
    const a = new QLearningAgent(d);
    a.totalEpisodes = d.totalEpisodes;
    a.wins   = d.wins;
    a.losses = d.losses;
    a.rating = d.rating;
    a.trainLog = d.trainLog || [];
    a.trained  = d.trained;

    // Podrska za stari dense format (d.Q) i novi sparse format (d.Qs)
    if (d.Qs) {
      a.Q = new Float32Array(STATE_SIZE * ACTIONS); // sve nule
      const {i: idxs, v: vals} = d.Qs;
      for (let k = 0; k < idxs.length; k++) a.Q[idxs[k]] = vals[k];
    } else if (d.Q) {
      a.Q = new Float32Array(d.Q); // backward compat
    }
    a.targetQ.set(a.Q);
    return a;
  }
}
