"""Real Spurgeon citations: pure-Python BM25 over his sermons, tagged with the
actual sermon title each passage came from.
retrieve_citations(query) -> HTML cards for the 'From His Own Pen' panel."""
import re, math, html as _html
from collections import Counter
from pathlib import Path

CORPUS = Path(__file__).parent / "data" / "spurgeon_clean.txt"
STOP = set("the a an of to and in is are be for that this with as it his he we our you your they "
           "them their him on by from not but or if so all who whom which what when where why how "
           "shall will would may can do does i me my mine us".split())
DELIM = "~ ~ ~ ~ ~ ~"

# Latin / Reformed / shorthand theology terms (and topics Spurgeon phrased differently)
# mapped to the plain English words that actually appear in his sermons.
SYNO = {
    # the five solas
    "sola fide": "faith alone justification justified righteousness believe",
    "sola gratia": "grace alone salvation free unmerited",
    "sola scriptura": "scripture word of God authority bible inspired",
    "solus christus": "Christ alone only mediator saviour",
    "soli deo gloria": "glory of God alone praise",
    # TULIP / doctrines of grace
    "tulip": "election predestination depravity grace perseverance saints sovereign",
    "total depravity": "dead in sin fallen corrupt sinful nature unable",
    "unconditional election": "chosen before foundation world predestinated sovereign",
    "limited atonement": "particular redemption sheep died for his people blood",
    "particular redemption": "limited atonement sheep died for his people blood",
    "irresistible grace": "effectual calling drawn quickened regenerated born again",
    "effectual call": "drawn quickened regenerated born again grace",
    "perseverance of the saints": "kept preserved eternal security final perseverance",
    "doctrines of grace": "election predestination grace sovereign calvinism",
    "calvinism": "election predestination sovereign grace doctrines",
    "arminianism": "free will man's choice falling away resist",
    "predestination": "election foreordained chosen before foundation sovereign decree",
    "election": "chosen predestinated foreordained sovereign before foundation",
    "reprobation": "vessels of wrath hardened passed by judgment",
    # soteriology
    "justification": "justified righteous faith imputed declared",
    "imputation": "righteousness imputed reckoned credited charged",
    "imputed righteousness": "righteousness reckoned credited Christ's obedience",
    "sanctification": "holiness growth grace holy mortify sin",
    "regeneration": "born again new birth quickened new heart",
    "adoption": "sons of God children heirs father adopted",
    "propitiation": "wrath appeased sacrifice atonement blood satisfied",
    "expiation": "sin atonement blood cleansed put away",
    "redemption": "redeemed ransom price blood bought purchased",
    "atonement": "blood sacrifice cross propitiation satisfaction substitute",
    "substitution": "substitute in our place bore our sins surety",
    "reconciliation": "reconciled peace with God enmity removed",
    "monergism": "grace alone God sovereign regeneration",
    "synergism": "free will cooperation man's part works",
    "assurance": "full assurance certain confidence know saved",
    "repentance": "repent turn from sin contrition sorrow godly",
    "conversion": "converted turned born again new creature",
    "faith": "believe trust believing reliance confidence",
    "grace": "free grace unmerited favour mercy",
    "covenant": "everlasting covenant promise covenant of grace",
    "covenant of grace": "everlasting covenant promise surety mediator",
    # the gospel / Christ
    "gospel": "good news glad tidings salvation cross Christ crucified",
    "incarnation": "Word made flesh God in flesh born virgin",
    "resurrection": "risen raised empty tomb living again life",
    "ascension": "ascended exalted right hand of God reigns",
    "second coming": "coming again return appearing judgment glory",
    "eschatology": "second coming judgment resurrection heaven hell last",
    "judgment": "great white throne day of judgment account wrath",
    "heaven": "glory rest mansions presence of God eternal life",
    "hell": "wrath fire punishment lost perish destruction",
    "trinity": "father son holy spirit godhead three persons",
    "holy spirit": "spirit comforter quicken sanctify indwelling",
    # life / pastoral
    "prayer": "pray supplication wrestling prayer intercession",
    "suffering": "affliction trial tribulation sorrow chastening",
    "depression": "downcast cast down despondent heavy spirit",
    "doubt": "unbelief weak faith fears questioning",
    "temptation": "tempted trial snare flesh devil resist",
    "sin": "transgression iniquity guilt corruption fallen",
    "law": "commandments moral law schoolmaster duty obedience",
    "works": "works of the law merit deeds righteousness own",
    "scripture": "word of God bible inspired infallible holy writ",
    "baptism": "baptized water buried symbol ordinance",
    "lord's supper": "communion table bread wine remembrance ordinance",
    "church": "body of Christ congregation saints assembly bride",
    "evangelism": "soul winning preach the gospel seek the lost",
    # other isms
    "universalism": "all men saved no hell everyone",
    "pelagianism": "free will man's own power no grace works",
    "antinomianism": "no law licence continue in sin grace abused",
    "legalism": "works righteousness self salvation merit law keeping",
    "papacy": "pope rome romish priestcraft antichrist",
    "purgatory": "rome romish after death cleansing fire invented",
}

def _expand(query):
    q = (query if isinstance(query, str) else str(query or "")).lower()
    extra = " ".join(v for k, v in SYNO.items() if k in q)
    return (query + " " + extra) if extra else query

def _tok(t):
    if not isinstance(t, str):
        t = " ".join(map(str, t)) if isinstance(t, (list, tuple)) else str(t or "")
    return re.findall(r"[a-z']+", t.lower())

def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _load_sermons():
    """Return list of (title, body). Corpus = TOC + sermon 1, then DELIM-separated
    sermons each beginning with their table-of-contents title."""
    raw = CORPUS.read_text(encoding="utf-8")
    head = raw.split(DELIM, 1)[0]
    toc = head.split("Table of Contents", 1)[-1]
    # TOC entries look like "<title> <number>." — capture the title before each number
    titles = [t.strip(" .\n") for t in re.findall(r"(.+?)\s+\d{1,3}\.", toc)]
    titles = [t for t in titles if 3 <= len(t) <= 90]
    # longest-first so prefix matching picks the most specific title
    index = sorted(((_norm(t), t) for t in titles), key=lambda x: -len(x[0]))

    sermons = []
    for seg in raw.split(DELIM)[1:]:          # skip part 0 (TOC + sermon 1, noisy)
        seg = seg.strip()
        nseg = _norm(seg[:140])
        title, body = "from his sermons", seg
        for ntitle, disp in index:
            if ntitle and nseg.startswith(ntitle):
                title = disp.replace("--", " — ")
                body = seg[len(disp):].lstrip(" .—-\"'\n")
                break
        sermons.append((title, body))
    return sermons

class _BM25:
    def __init__(self, passages, titles):
        self.passages, self.titles = passages, titles
        self.docs = [_tok(p) for p in passages]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        df = Counter()
        for d in self.docs:
            for w in set(d): df[w] += 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def top(self, query, k=5, k1=1.5, b=0.75):
        qt = [w for w in _tok(query) if w not in STOP and w in self.idf]
        if not qt:
            return []
        scores = [0.0] * self.N
        for w in qt:
            idf = self.idf[w]
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    dl = len(self.docs[i])
                    scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / self.avgdl))
        ranked = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        return [(self.passages[i], self.titles[i], scores[i]) for i in ranked[:k] if scores[i] > 0]

def _build():
    if not CORPUS.exists():
        return None
    passages, titles = [], []
    for title, body in _load_sermons():
        sents = [s.strip() for s in body.splitlines() if len(s.strip()) > 20]
        for i in range(0, len(sents), 4):
            p = " ".join(sents[i:i + 4])
            if 30 <= len(p.split()) <= 130:
                passages.append(p)
                titles.append(title)
    return _BM25(passages, titles) if passages else None

_INDEX = _build()

def _trim(p, n=48):
    w = p.split()
    return p if len(w) <= n else " ".join(w[:n]).rstrip(",;:") + "…"

EMPTY_HTML = "<div class='pen-empty'>Ask a question and his own words will appear here.</div>"
PEN_EMPTY = EMPTY_HTML

def retrieve_citations(query, k=5):
    """Return HTML cards of the top real Spurgeon passages, each with its sermon title."""
    if _INDEX is None:
        return "<div class='pen-empty'>Corpus not loaded.</div>"
    hits = _INDEX.top(_expand(query), k=k)
    if not hits:
        return "<div class='pen-empty'>No closely matching passage found.</div>"
    cards = []
    for passage, title, _ in hits:
        quote = _html.escape(_trim(passage))
        src = _html.escape(title)
        cards.append(
            "<div class='pen-card'>"
            f"<div class='pen-quote'>“{quote}”</div>"
            f"<div class='pen-src'>C. H. Spurgeon · <em>{src}</em></div>"
            "</div>"
        )
    return "<div class='pen-list'>" + "".join(cards) + "</div>"
