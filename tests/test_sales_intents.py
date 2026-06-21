from app.sales_intents import parse_sales_intent


def test_today_revenue() -> None:
    intent = parse_sales_intent("今天各门店的营业额是多少？")
    assert intent is not None
    assert intent.metric == "store_revenue"
    assert intent.period == "today"


def test_week_qty() -> None:
    intent = parse_sales_intent("本周哪家店销售件数最多？")
    assert intent is not None
    assert intent.metric == "store_qty"
    assert intent.period == "this_week"


def test_month_top_amount() -> None:
    intent = parse_sales_intent("本月销售最好的商品是哪几款？")
    assert intent is not None
    assert intent.metric == "top_amount"
    assert intent.period == "this_month"
    assert intent.top_n >= 1


def test_quarter_hot_product() -> None:
    intent = parse_sales_intent("本季度爆款是哪几款？")
    assert intent is not None
    assert intent.metric == "top_qty"
    assert intent.period == "this_quarter"


def test_year_top_qty_explicit_n() -> None:
    intent = parse_sales_intent("今年卖得最多的前 3 款商品")
    assert intent is not None
    assert intent.period == "this_year"
    assert intent.metric == "top_qty"
    assert intent.top_n == 3


def test_singular_hot_product_defaults_to_top_one() -> None:
    intent = parse_sales_intent("昨天爆款是哪一件？")
    assert intent is not None
    assert intent.period == "yesterday"
    assert intent.metric == "top_qty"
    assert intent.top_n == 1


def test_store_context_beats_top_qty_keywords() -> None:
    intent = parse_sales_intent("上个月哪家店卖得最多件")
    assert intent is not None
    assert intent.metric == "store_qty"
    assert intent.period == "last_month"


def test_yesterday_revenue() -> None:
    intent = parse_sales_intent("昨天卖了多少钱？")
    assert intent is not None
    assert intent.period == "yesterday"
    assert intent.metric == "store_revenue"


def test_non_sales_returns_none() -> None:
    assert parse_sales_intent("帮我润色一下这段话") is None
    assert parse_sales_intent("今天天气怎么样") is None


def test_blank_input() -> None:
    assert parse_sales_intent("") is None
    assert parse_sales_intent("   ") is None
