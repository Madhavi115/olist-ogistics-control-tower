
"""Send five sample orders to the Olist investigation API."""

import json
from urllib.request import Request, urlopen

orders = [
    ("ORDER-001", "2018-03-01T10:00:00",
     "2018-03-20T00:00:00", 2, 150, 20, "SP"),
    ("ORDER-002", "2018-03-02T09:00:00",
     "2018-03-08T00:00:00", 4, 450, 60, "AM"),
    ("ORDER-003", "2018-03-03T15:00:00",
     "2018-04-01T00:00:00", 1, 50, 10, "RJ"),
    ("ORDER-004", "2018-03-04T11:00:00",
     "2018-03-12T00:00:00", 3, 300, 45, "BA"),
    ("ORDER-005", "2018-03-05T08:00:00",
     "2018-03-30T00:00:00", 1, 90, 15, "MG"),
]

payload = {
    "top_fraction": 0.1,
    "orders": [
        {
            "order_id": order_id,
            "purchase_timestamp": purchase,
            "estimated_delivery_date": estimate,
            "item_count": count,
            "total_price": price,
            "total_freight": freight,
            "customer_state": state,
        }
        for order_id, purchase, estimate, count, price, freight, state
        in orders
    ],
}

request = Request(
    "http://127.0.0.1:8000/investigate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

print("Sending 5 orders to the investigation API...")

with urlopen(request, timeout=180) as response:
    result = json.load(response)

print("Orders analyzed:", result["ranking"]["total_orders"])
print("Priority count:", result["ranking"]["priority_count"])
print("AI status:", result["ai_status"])

for shipment in result["ranking"]["ranked_shipments"]:
    print(
        shipment["rank"],
        shipment["order_id"],
        shipment["delay_risk_score"],
        shipment["priority_investigation"],
    )

print("\nAI analysis:")
print(json.dumps(result["ai_analysis"], indent=2))
