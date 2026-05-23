# Karpathy Guidelines — Examples

Practical examples of the four principles applied to real coding scenarios.
Source: https://github.com/multica-ai/andrej-karpathy-skills

---

## 1. Think Before Coding

**Scenario:** User asks — "Export user data."

**Wrong (silent assumption):**
```python
# Just picks CSV, all fields, all users — no clarification
def export_users():
    return User.objects.all().to_csv()
```

**Right (surface the assumptions first):**
> Before implementing, ask:
> - Scope: all users, or filtered (active only, date range)?
> - Format: file download, API response, or email attachment?
> - Fields: all fields, or exclude PII / sensitive data?
> - Volume: is this a one-off or scheduled export?

Only implement once the answers are known. A wrong assumption here produces the wrong feature — not a bug, which is harder to detect and correct.

---

## 2. Simplicity First

**Scenario:** Build a discount calculator.

**Wrong (premature abstraction):**
```python
class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, price: float) -> float: ...

class PercentageDiscount(DiscountStrategy):
    def __init__(self, pct): self.pct = pct
    def calculate(self, price): return price * (1 - self.pct)

class DiscountFactory:
    _registry: dict[str, type[DiscountStrategy]] = {}
    ...
```

**Right (solve today's problem):**
```python
def apply_discount(price: float, discount_pct: float) -> float:
    return price * (1 - discount_pct)
```

Add the strategy pattern only when a second discount type actually exists. "Solve today's problem simply, not tomorrow's problem prematurely."

---

## 3. Surgical Changes

**Scenario:** Fix a bug — empty string passes email validation.

**Wrong (touching unrelated code):**
```python
# Fixed the bug AND reformatted quotes, added type hints,
# restructured the condition, renamed the variable
def validate_email(email: str) -> bool:
    if not email or not email.strip():  # fixed
        return False
    return bool(EMAIL_REGEX.match(email.strip()))  # reformatted
```

**Right (change only the broken lines):**
```python
def validate_email(email):
    if not email:  # was: if email is None
        return False
    return bool(EMAIL_REGEX.match(email))
```

Every changed line must trace directly to the reported bug. The reformatting and type hint are separate tasks — mention them, don't do them unprompted.

---

## 4. Goal-Driven Execution

**Scenario:** "Review the auth module and improve it."

**Wrong (vague plan):**
> I'll review the auth module and make improvements where I see fit.

**Right (verifiable steps):**
```
1. Read auth/session.py and auth/tokens.py → verify: understand current flow
2. Write a test reproducing the reported session-expiry race condition → verify: test fails
3. Fix the race condition → verify: test passes
4. Run full auth test suite → verify: no regressions (all existing tests green)
5. Note (but do not fix) unrelated dead code spotted in auth/legacy.py
```

Each step has a concrete check. If step 3 fails, you know exactly where to loop. Vague goals ("make it work") require constant clarification and produce unpredictable diffs.

---

## Core Insight

Overcomplicated code often follows legitimate design patterns — it just arrives **prematurely**. The question is not "is this pattern valid?" but "does today's requirement justify this complexity?"

Simpler code:
- Has fewer bugs (less surface area)
- Is easier to understand and review
- Can be refactored when genuine complexity emerges
- Produces smaller, more reviewable diffs
