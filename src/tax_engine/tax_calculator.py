from decimal import Decimal, ROUND_HALF_UP


MONEY_PRECISION = Decimal("0.01")


def to_decimal(value) -> Decimal:
    if isinstance(value, Decimal):
        return value

    return Decimal(str(value))


def calculate_tax(
    taxable_amount,
    tax_rate,
) -> Decimal:
    """
    Calculate tax from taxable amount and percentage rate.
    """

    amount = to_decimal(taxable_amount)
    rate = to_decimal(tax_rate)

    tax = amount * rate / Decimal("100")

    return tax.quantize(
        MONEY_PRECISION,
        rounding=ROUND_HALF_UP,
    )


def calculate_difference(
    declared_tax,
    expected_tax,
) -> Decimal:

    declared = to_decimal(declared_tax)
    expected = to_decimal(expected_tax)

    return (
        declared - expected
    ).quantize(
        MONEY_PRECISION,
        rounding=ROUND_HALF_UP,
    )
