"""Search actual generated PDF pages with local lexical retrieval; cite page and hash."""

import argparse
import json
import math
import re

from .pdf_knowledge import chunks



def search(query: str, limit: int = 4):
    documents = chunks()
    if not query.strip():
        return documents[:limit]
    aliases = {"气": " pneumatic pressure air", "漏": " leak leakage", "维护": " maintenance",
               "手机": " smartphone housing", "翻转": " turning reorientation", "皮带": " conveyor",
               "故障": " fault", "订单": " order", "实例": " instance"}
    expanded = query.lower() + "".join(value for key, value in aliases.items() if key in query)
    terms = set(re.findall(r"\w+", expanded))
    tokens = [re.findall(r"\w+", (doc["title"] + " " + doc["text"]).lower()) for doc in documents]
    ranked = []
    for doc, words in zip(documents, tokens):
        score = sum(
            (1 + math.log(words.count(term))) * math.log(1 + len(documents) / sum(term in t for t in tokens))
            for term in terms if term in words
        )
        if score:
            ranked.append({**doc, "score": round(score, 4)})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="pressure leakage maintenance")
    parser.add_argument("--limit", type=int, choices=range(1, 8), default=4)
    args = parser.parse_args()
    print(json.dumps({"retrieval": "PDF text / local lexical", "query": args.query,
                      "instruction": "Cite PDF filename, page, chunk ID and SHA-256; distinguish reference facts from demo fiction.",
                      "chunks": search(args.query, args.limit)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
