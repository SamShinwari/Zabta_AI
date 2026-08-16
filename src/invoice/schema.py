from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Invoice:

    # --------------------------------------------------------
    # Invoice Identification
    # --------------------------------------------------------

    invoice_number: str
    invoice_date: date

    # --------------------------------------------------------
    # Parties
    # --------------------------------------------------------

    supplier_strn_ntn_cnic: str
    buyer_strn_ntn_cnic: str

    # --------------------------------------------------------
    # Transaction Classification
    # --------------------------------------------------------

    invoice_type: str
    purchase_type: str

    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    item_description: str
    hs_code: str

    # --------------------------------------------------------
    # Amounts / Quantities
    # --------------------------------------------------------

    quantity: float
    unit_price: float

    value_excluding_sales_tax: float

    # --------------------------------------------------------
    # Declared Sales Tax
    # --------------------------------------------------------

    sales_tax_rate: Optional[float]
    sales_tax_amount: Optional[float]