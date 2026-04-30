"""
Лабораторна робота 11 — Варіант 8: Ресторан з доставкою
Завдання: класифікація замовлень за типом кухні, витяг страв/кількостей, виявлення спеціальних побажань.
"""

import os
import json
import time
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-oss-120b:free"  # безкоштовна модель

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ─── JSON-схема виходу ────────────────────────────────────────────────────────
CUISINE_TYPES = ["ukrainian", "european", "asian", "fastfood", "desserts"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cuisine_type": {
            "type": "string",
            "enum": CUISINE_TYPES,
            "description": "Тип кухні замовлення"
        },
        "dishes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1}
                },
                "required": ["name", "quantity"]
            },
            "description": "Список страв із кількостями"
        },
        "special_requests": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Спеціальні побажання (без цибулі, гостро тощо)"
        }
    },
    "required": ["cuisine_type", "dishes", "special_requests"]
}


# ─── Допоміжні функції ────────────────────────────────────────────────────────

def call_api(messages: list, temperature: float = 0, retries: int = 5) -> dict:
    """Надсилає запит до OpenRouter з автоматичним retry при 429."""
    payload = {
        "model": MODEL,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    for attempt in range(retries):
        t0 = time.time()
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        elapsed = time.time() - t0

        if resp.status_code == 429:
            wait = 15 * (attempt + 1)  # 15s, 30s, 45s ...
            print(f"       ⏳ Rate limit (429), чекаємо {wait}с...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "content": json.loads(content),
            "elapsed": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    raise RuntimeError("Вичерпано всі спроби (retry) через rate limit 429")


def validate_output(obj: dict) -> bool:
    """Перевірка структури витягнутого об'єкта."""
    if not isinstance(obj, dict):
        return False
    if obj.get("cuisine_type") not in CUISINE_TYPES:
        return False
    if not isinstance(obj.get("dishes"), list) or len(obj["dishes"]) == 0:
        return False
    for d in obj["dishes"]:
        if not isinstance(d.get("name"), str) or not isinstance(d.get("quantity"), int):
            return False
    if not isinstance(obj.get("special_requests"), list):
        return False
    return True


def load_json(path: str) -> any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Збережено: {path}")


# ─── Промпти ──────────────────────────────────────────────────────────────────

SYSTEM_BASE = (
    "Ти — система аналізу замовлень ресторану з доставкою. "
    "Відповідай ЛИШЕ валідним JSON без будь-якого додаткового тексту. "
    "Схема JSON: {cuisine_type: string (ukrainian|european|asian|fastfood|desserts), "
    "dishes: [{name: string, quantity: integer}], "
    "special_requests: [string]}."
)

FEW_SHOT_EXAMPLES = """
Приклад 1:
Вхід: "Замовляю роли Каліфорнія 24 шт та місо-суп 1 порція. Без васабі."
Вихід: {"cuisine_type":"asian","dishes":[{"name":"роли Каліфорнія","quantity":24},{"name":"місо-суп","quantity":1}],"special_requests":["без васабі"]}

Приклад 2:
Вхід: "Дайте 2 гамбургери з сиром і велику картоплю фрі 1 шт. Без кетчупу."
Вихід: {"cuisine_type":"fastfood","dishes":[{"name":"гамбургер з сиром","quantity":2},{"name":"картопля фрі велика","quantity":1}],"special_requests":["без кетчупу"]}

Приклад 3:
Вхід: "Потрібен штрудель яблучний 2 шт і морозиво ванільне 3 кульки."
Вихід: {"cuisine_type":"desserts","dishes":[{"name":"штрудель яблучний","quantity":2},{"name":"морозиво ванільне","quantity":3}],"special_requests":[]}
"""

COT_INSTRUCTION = (
    "Перед тим, як дати відповідь, міркуй крок за кроком у полі 'reasoning': "
    "1) визнач тип кухні, 2) витягни всі страви з кількостями, 3) знайди спеціальні побажання. "
    "Включи поле 'reasoning' у JSON-відповідь разом з іншими полями."
)


# ─── Чотири стратегії ─────────────────────────────────────────────────────────

def zero_shot(text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": f"Проаналізуй замовлення:\n{text}"},
    ]
    return call_api(messages, temperature=0)


def few_shot(text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": (
            f"Ось приклади правильної розмітки:\n{FEW_SHOT_EXAMPLES}\n"
            f"Тепер проаналізуй замовлення:\n{text}"
        )},
    ]
    return call_api(messages, temperature=0)


def chain_of_thought(text: str) -> dict:
    system = SYSTEM_BASE + " " + COT_INSTRUCTION
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Проаналізуй замовлення крок за кроком:\n{text}"},
    ]
    return call_api(messages, temperature=0)


def self_consistency(text: str, n: int = 5) -> dict:
    """N незалежних CoT-прогонів → majority voting за кожним полем."""
    system = SYSTEM_BASE + " " + COT_INSTRUCTION
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Проаналізуй замовлення крок за кроком:\n{text}"},
    ]
    runs = []
    total_elapsed = 0
    total_prompt = 0
    total_completion = 0
    for _ in range(n):
        r = call_api(messages, temperature=0.8)
        runs.append(r["content"])
        total_elapsed += r["elapsed"]
        total_prompt += r["prompt_tokens"]
        total_completion += r["completion_tokens"]
        time.sleep(8)  # rate limit між SC-прогонами

    # Majority voting по cuisine_type
    cuisine_votes = Counter(r.get("cuisine_type") for r in runs if r.get("cuisine_type") in CUISINE_TYPES)
    best_cuisine = cuisine_votes.most_common(1)[0][0] if cuisine_votes else "fastfood"

    # Majority voting по dishes: беремо найчастіший набір як tuple
    def dishes_key(d_list):
        return tuple(sorted((d.get("name", ""), d.get("quantity", 1)) for d in d_list))

    dishes_counter = Counter(dishes_key(r.get("dishes", [])) for r in runs)
    best_dishes_key = dishes_counter.most_common(1)[0][0]
    # Відновлюємо список зі збереженого ключа
    best_dishes = [{"name": n, "quantity": q} for n, q in best_dishes_key]

    # Majority voting по special_requests: зберегти ті, що є у ≥ n/2 прогонів
    all_requests = []
    for r in runs:
        all_requests.extend(r.get("special_requests", []))
    req_counter = Counter(all_requests)
    best_requests = [req for req, cnt in req_counter.items() if cnt >= n / 2]

    voted = {
        "cuisine_type": best_cuisine,
        "dishes": best_dishes,
        "special_requests": best_requests,
    }
    return {
        "content": voted,
        "elapsed": round(total_elapsed, 2),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
    }


# ─── Accuracy ─────────────────────────────────────────────────────────────────

def compute_accuracy(results: dict, gold: dict) -> dict:
    ids = [str(k) for k in gold.keys()]
    cuisine_correct = 0
    dishes_correct = 0
    requests_correct = 0
    valid_count = 0

    for sid in ids:
        pred = results.get(sid, {})
        ref = gold[sid]
        if not pred:
            continue
        valid_count += 1

        # cuisine_type
        if pred.get("cuisine_type") == ref.get("cuisine_type"):
            cuisine_correct += 1

        # dishes: порівняння за назвою (fuzzy: перевіряємо входження)
        pred_names = {d["name"].lower().strip() for d in pred.get("dishes", [])}
        gold_names = {d["name"].lower().strip() for d in ref.get("dishes", [])}
        if pred_names == gold_names:
            dishes_correct += 1

        # special_requests: хоч одне збігається або обидва порожні
        pred_reqs = set(r.lower().strip() for r in pred.get("special_requests", []))
        gold_reqs = set(r.lower().strip() for r in ref.get("special_requests", []))
        if pred_reqs == gold_reqs or (not pred_reqs and not gold_reqs):
            requests_correct += 1

    n = len(ids)
    return {
        "accuracy_cuisine": round(cuisine_correct / n, 2),
        "accuracy_dishes": round(dishes_correct / n, 2),
        "accuracy_requests": round(requests_correct / n, 2),
        "accuracy_overall": round((cuisine_correct + dishes_correct + requests_correct) / (3 * n), 2),
    }


# ─── Головна функція ──────────────────────────────────────────────────────────

def run_strategy(name: str, fn, dataset: list) -> tuple[dict, dict]:
    print(f"\n{'=' * 50}")
    print(f"  Стратегія: {name.upper()}")
    print(f"{'=' * 50}")

    results = {}
    stats = {"prompt_tokens": 0, "completion_tokens": 0, "elapsed": 0, "valid": 0}

    for record in dataset:
        sid = str(record["id"])
        print(f"  [{sid:>2}] {record['text'][:60]}...")
        try:
            r = fn(record["text"])
            content = r["content"]
            if validate_output(content):
                results[sid] = content
                stats["valid"] += 1
            else:
                print(f"       ⚠ Валідація не пройшла: {content}")
                results[sid] = {}
            stats["prompt_tokens"] += r["prompt_tokens"]
            stats["completion_tokens"] += r["completion_tokens"]
            stats["elapsed"] += r["elapsed"]
        except Exception as e:
            print(f"       ✗ Помилка: {e}")
            results[sid] = {}
        time.sleep(8)  # ~7 req/хв — безпечно для безкоштовного тарифу

    n = len(dataset)
    stats["valid_rate"] = round(stats["valid"] / n * 100, 1)
    stats["avg_prompt_tokens"] = round(stats["prompt_tokens"] / n, 0)
    stats["avg_completion_tokens"] = round(stats["completion_tokens"] / n, 0)
    stats["avg_elapsed"] = round(stats["elapsed"] / n, 2)
    stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]

    save_json(f"results_{name}.json", results)
    return results, stats


def main():
    dataset = load_json("dataset.json")
    gold = load_json("gold_labels.json")

    strategies = {
        "zero_shot": zero_shot,
        "few_shot": few_shot,
        "cot": chain_of_thought,
        "self_consistency": self_consistency,
    }

    all_metrics = {}

    for name, fn in strategies.items():
        results, stats = run_strategy(name, fn, dataset)
        acc = compute_accuracy(results, gold)
        all_metrics[name] = {**stats, **acc}

    # ─── Зберегти метрики ──────────────────────────────────────────────────
    save_json("metrics.json", all_metrics)

    # ─── Порівняльна таблиця ────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ПОРІВНЯЛЬНА ТАБЛИЦЯ МЕТРИК")
    print("=" * 80)
    header = f"{'Метрика':<30} {'Zero-shot':>13} {'Few-shot':>13} {'CoT':>13} {'Self-Cons':>13}"
    print(header)
    print("-" * 80)

    rows = [
        ("Valid JSON, %", "valid_rate", "{:.1f}"),
        ("Accuracy (cuisine_type)", "accuracy_cuisine", "{:.2f}"),
        ("Accuracy (dishes)", "accuracy_dishes", "{:.2f}"),
        ("Accuracy (spec_requests)", "accuracy_requests", "{:.2f}"),
        ("Accuracy (загальна)", "accuracy_overall", "{:.2f}"),
        ("Сер. токенів (вхід)", "avg_prompt_tokens", "{:.0f}"),
        ("Сер. токенів (вихід)", "avg_completion_tokens", "{:.0f}"),
        ("Сер. час запиту, с", "avg_elapsed", "{:.2f}"),
        ("Всього токенів", "total_tokens", "{:.0f}"),
    ]
    cols = ["zero_shot", "few_shot", "cot", "self_consistency"]
    for label, key, fmt in rows:
        vals = [fmt.format(all_metrics[c].get(key, 0)) for c in cols]
        print(f"{label:<30} {vals[0]:>13} {vals[1]:>13} {vals[2]:>13} {vals[3]:>13}")

    print("=" * 80)
    print("\nГотово! Файли збережено у поточній директорії.")


if __name__ == "__main__":
    main()
