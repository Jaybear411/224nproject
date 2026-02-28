#!/usr/bin/env python3
"""Generate a diverse tool-calling training dataset."""
import json
import random
from pathlib import Path

SEED = 11711

TOOLS = {
  "get_weather": {
    "schema": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
      }
    },
    "templates": [
      ("What's the weather in {city}?", {"unit": "fahrenheit"}),
      ("Weather in {city} please.", {"unit": "fahrenheit"}),
      ("Get me the weather for {city} in {unit}.", {}),
      ("How's the weather in {city} right now?", {"unit": "fahrenheit"}),
      ("Tell me the current weather in {city} ({unit}).", {}),
      ("Can you check the weather in {city}?", {"unit": "fahrenheit"}),
      ("I want to know the {unit} temperature in {city}.", {}),
      ("Show me weather for {city} in {unit}.", {}),
      ("What is the temperature in {city}?", {"unit": "fahrenheit"}),
      ("Check weather conditions in {city}, use {unit}.", {}),
    ],
    "slots": {
      "city": [
        "San Francisco", "New York", "Tokyo", "London", "Paris",
        "Berlin", "Mumbai", "Sydney", "Toronto", "Seoul",
        "Chicago", "Los Angeles", "Houston", "Phoenix", "Seattle",
        "Denver", "Boston", "Atlanta", "Miami", "Dallas",
        "Rome", "Madrid", "Amsterdam", "Vienna", "Prague",
        "Bangkok", "Singapore", "Dubai", "Cairo", "Lagos",
      ],
      "unit": ["celsius", "fahrenheit"],
    }
  },
  "book_flight": {
    "schema": {
      "name": "book_flight",
      "description": "Book a one-way flight",
      "parameters": {
        "type": "object",
        "properties": {
          "origin": {"type": "string"},
          "destination": {"type": "string"},
          "date": {"type": "string"}
        },
        "required": ["origin", "destination", "date"]
      }
    },
    "templates": [
      ("Book a flight from {origin} to {destination} on {date}.", {}),
      ("I need to fly from {origin} to {destination} on {date}.", {}),
      ("Reserve a one-way ticket from {origin} to {destination} for {date}.", {}),
      ("Get me a flight {origin} to {destination}, {date}.", {}),
      ("Can you book a flight departing {origin} arriving {destination} on {date}?", {}),
      ("Please arrange a flight from {origin} to {destination} on {date}.", {}),
      ("I want to travel from {origin} to {destination} on {date}.", {}),
      ("Flying from {origin} to {destination} on {date}, book it.", {}),
    ],
    "slots": {
      "origin": [
        "SFO", "JFK", "LAX", "ORD", "ATL", "DFW", "SEA", "BOS",
        "MIA", "DEN", "London", "Paris", "Berlin", "Tokyo", "Delhi",
      ],
      "destination": [
        "JFK", "LAX", "SFO", "ORD", "LHR", "CDG", "NRT", "BLR",
        "DXB", "SIN", "Rome", "Dublin", "Lisbon", "Miami", "Denver",
      ],
      "date": [
        "2026-03-15", "2026-04-02", "2026-04-18", "2026-05-01",
        "2026-05-21", "2026-06-08", "2026-07-01", "2026-07-14",
        "2026-08-18", "2026-09-12", "2026-10-05", "2026-11-20",
      ],
    }
  },
  "send_email": {
    "schema": {
      "name": "send_email",
      "description": "Send an email to a recipient",
      "parameters": {
        "type": "object",
        "properties": {
          "to": {"type": "string"},
          "subject": {"type": "string"},
          "body": {"type": "string"}
        },
        "required": ["to", "subject", "body"]
      }
    },
    "templates": [
      ("Send an email to {to} with subject \"{subject}\" saying \"{body}\".", {}),
      ("Email {to}: subject is \"{subject}\", body is \"{body}\".", {}),
      ("Draft an email to {to} about \"{subject}\" with message \"{body}\".", {}),
      ("Compose a message to {to}, subject \"{subject}\", content \"{body}\".", {}),
      ("Write an email to {to} titled \"{subject}\" that says \"{body}\".", {}),
      ("Please send {to} an email. Subject: \"{subject}\". Body: \"{body}\".", {}),
    ],
    "slots": {
      "to": [
        "alice@example.com", "bob@company.org", "carol@mail.com",
        "dave@startup.io", "eve@university.edu", "frank@work.com",
        "grace@team.org", "henry@corp.net",
      ],
      "subject": [
        "Meeting Tomorrow", "Project Update", "Quick Question",
        "Follow Up", "Schedule Change", "Weekly Report",
        "Lunch Plans", "Deadline Reminder",
      ],
      "body": [
        "Can we meet at 3pm?", "The project is on track.",
        "Do you have time to chat?", "Just following up on our call.",
        "The meeting has been moved to Friday.",
        "Please see the attached report.", "Are you free for lunch?",
        "Please submit by end of day.",
      ],
    }
  },
  "create_reminder": {
    "schema": {
      "name": "create_reminder",
      "description": "Create a reminder with a message and time",
      "parameters": {
        "type": "object",
        "properties": {
          "message": {"type": "string"},
          "time": {"type": "string"}
        },
        "required": ["message", "time"]
      }
    },
    "templates": [
      ("Remind me to {message} at {time}.", {}),
      ("Set a reminder: {message} at {time}.", {}),
      ("Create a reminder for {time} to {message}.", {}),
      ("I need a reminder at {time}: {message}.", {}),
      ("Don't let me forget to {message} at {time}.", {}),
      ("Please remind me at {time} to {message}.", {}),
    ],
    "slots": {
      "message": [
        "call the dentist", "pick up groceries", "submit the report",
        "take medication", "water the plants", "check the oven",
        "join the meeting", "send the invoice", "walk the dog",
        "review the PR", "call mom", "pay rent",
      ],
      "time": [
        "9:00 AM", "10:30 AM", "12:00 PM", "2:00 PM",
        "3:30 PM", "5:00 PM", "6:00 PM", "8:00 PM",
      ],
    }
  },
  "search_restaurants": {
    "schema": {
      "name": "search_restaurants",
      "description": "Search for restaurants by cuisine and location",
      "parameters": {
        "type": "object",
        "properties": {
          "cuisine": {"type": "string"},
          "location": {"type": "string"},
          "price_range": {"type": "string", "enum": ["low", "medium", "high"]}
        },
        "required": ["cuisine", "location"]
      }
    },
    "templates": [
      ("Find {cuisine} restaurants in {location}.", {"price_range": "medium"}),
      ("Search for {price_range}-priced {cuisine} food near {location}.", {}),
      ("I want {cuisine} in {location}, {price_range} budget.", {}),
      ("Show me {cuisine} places in {location}.", {"price_range": "medium"}),
      ("Any good {cuisine} restaurants in {location}? Budget is {price_range}.", {}),
      ("Look up {cuisine} dining options in {location}.", {"price_range": "medium"}),
    ],
    "slots": {
      "cuisine": [
        "Italian", "Japanese", "Mexican", "Indian", "Thai",
        "Chinese", "French", "Korean", "Vietnamese", "Greek",
      ],
      "location": [
        "downtown", "midtown", "the west side", "near campus",
        "the financial district", "SoHo", "Palo Alto",
        "Mountain View", "the Marina", "Capitol Hill",
      ],
      "price_range": ["low", "medium", "high"],
    }
  },
  "translate_text": {
    "schema": {
      "name": "translate_text",
      "description": "Translate text from one language to another",
      "parameters": {
        "type": "object",
        "properties": {
          "text": {"type": "string"},
          "source_language": {"type": "string"},
          "target_language": {"type": "string"}
        },
        "required": ["text", "target_language"]
      }
    },
    "templates": [
      ("Translate \"{text}\" to {target_language}.", {"source_language": "English"}),
      ("How do you say \"{text}\" in {target_language}?", {"source_language": "English"}),
      ("Convert \"{text}\" from {source_language} to {target_language}.", {}),
      ("Translate the following to {target_language}: \"{text}\".", {"source_language": "English"}),
      ("I need \"{text}\" translated into {target_language}.", {"source_language": "English"}),
      ("What is \"{text}\" in {target_language}?", {"source_language": "English"}),
    ],
    "slots": {
      "text": [
        "Hello, how are you?", "Where is the train station?",
        "I would like a coffee please.", "Thank you very much.",
        "What time does the store close?", "How much does this cost?",
        "Nice to meet you.", "Can you help me?",
        "I am lost.", "The weather is nice today.",
      ],
      "source_language": ["English", "Spanish", "French", "German"],
      "target_language": [
        "Spanish", "French", "German", "Japanese",
        "Mandarin", "Portuguese", "Italian", "Korean",
      ],
    }
  },
  "set_alarm": {
    "schema": {
      "name": "set_alarm",
      "description": "Set an alarm for a specific time",
      "parameters": {
        "type": "object",
        "properties": {
          "time": {"type": "string"},
          "label": {"type": "string"}
        },
        "required": ["time"]
      }
    },
    "templates": [
      ("Set an alarm for {time}.", {"label": "Alarm"}),
      ("Wake me up at {time}.", {"label": "Wake up"}),
      ("Set a {time} alarm labeled \"{label}\".", {}),
      ("I need an alarm at {time} called \"{label}\".", {}),
      ("Alarm at {time} please.", {"label": "Alarm"}),
      ("Create an alarm for {time}, name it \"{label}\".", {}),
    ],
    "slots": {
      "time": [
        "6:00 AM", "6:30 AM", "7:00 AM", "7:30 AM", "8:00 AM",
        "8:30 AM", "9:00 AM", "5:00 AM", "5:30 AM", "10:00 AM",
      ],
      "label": [
        "Wake up", "Morning workout", "Standup meeting",
        "Take pills", "School run", "Gym time",
        "Early flight", "Work",
      ],
    }
  },
  "get_stock_price": {
    "schema": {
      "name": "get_stock_price",
      "description": "Get the current stock price for a ticker symbol",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {"type": "string"},
          "exchange": {"type": "string"}
        },
        "required": ["ticker"]
      }
    },
    "templates": [
      ("What's the stock price of {ticker}?", {"exchange": "NYSE"}),
      ("Get me the price for {ticker}.", {"exchange": "NYSE"}),
      ("How is {ticker} doing today?", {"exchange": "NYSE"}),
      ("Check the current price of {ticker} on {exchange}.", {}),
      ("Look up {ticker} stock price.", {"exchange": "NYSE"}),
      ("Show me {ticker} on {exchange}.", {}),
    ],
    "slots": {
      "ticker": [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META",
        "NVDA", "JPM", "V", "WMT", "DIS", "NFLX",
      ],
      "exchange": ["NYSE", "NASDAQ", "LSE"],
    }
  },
  "convert_currency": {
    "schema": {
      "name": "convert_currency",
      "description": "Convert an amount from one currency to another",
      "parameters": {
        "type": "object",
        "properties": {
          "amount": {"type": "number"},
          "from_currency": {"type": "string"},
          "to_currency": {"type": "string"}
        },
        "required": ["amount", "from_currency", "to_currency"]
      }
    },
    "templates": [
      ("Convert {amount} {from_currency} to {to_currency}.", {}),
      ("How much is {amount} {from_currency} in {to_currency}?", {}),
      ("What's {amount} {from_currency} worth in {to_currency}?", {}),
      ("{amount} {from_currency} to {to_currency} please.", {}),
      ("Exchange {amount} {from_currency} into {to_currency}.", {}),
      ("I need to convert {amount} {from_currency} to {to_currency}.", {}),
    ],
    "slots": {
      "amount": [10, 25, 50, 100, 200, 500, 1000, 2500, 5000, 75, 150, 350],
      "from_currency": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"],
      "to_currency": ["EUR", "USD", "GBP", "JPY", "INR", "CHF", "CAD"],
    }
  },
  "calculate_tip": {
    "schema": {
      "name": "calculate_tip",
      "description": "Calculate tip amount for a bill",
      "parameters": {
        "type": "object",
        "properties": {
          "bill_amount": {"type": "number"},
          "tip_percentage": {"type": "number"},
          "split": {"type": "integer"}
        },
        "required": ["bill_amount", "tip_percentage"]
      }
    },
    "templates": [
      ("Calculate a {tip_percentage}% tip on a ${bill_amount} bill.", {"split": 1}),
      ("What's the tip on ${bill_amount} at {tip_percentage} percent?", {"split": 1}),
      ("Tip for ${bill_amount}, {tip_percentage}%, split {split} ways.", {}),
      ("How much should I tip on ${bill_amount} at {tip_percentage}%?", {"split": 1}),
      ("{tip_percentage}% tip on ${bill_amount}, divided by {split} people.", {}),
      ("Bill is ${bill_amount}. {tip_percentage}% tip, split {split} ways.", {}),
    ],
    "slots": {
      "bill_amount": [25.50, 42.00, 67.80, 89.99, 120.00, 35.75, 55.00, 150.00, 200.00, 18.50],
      "tip_percentage": [15, 18, 20, 22, 25],
      "split": [1, 2, 3, 4, 5, 6],
    }
  },
}

TARGET_COUNT = 300


def generate_examples(rng):
  examples = []
  idx = 0

  tool_names = list(TOOLS.keys())
  per_tool = TARGET_COUNT // len(tool_names)

  for tool_name in tool_names:
    tool = TOOLS[tool_name]
    schema = tool["schema"]
    templates = tool["templates"]
    slots = tool["slots"]

    generated_for_tool = 0
    attempts = 0
    seen_args = set()

    while generated_for_tool < per_tool and attempts < per_tool * 10:
      attempts += 1
      tmpl_text, defaults = rng.choice(templates)

      filled_slots = {}
      for slot_name, slot_values in slots.items():
        filled_slots[slot_name] = rng.choice(slot_values)

      if tool_name == "book_flight" and filled_slots.get("origin") == filled_slots.get("destination"):
        continue

      if tool_name == "convert_currency" and filled_slots.get("from_currency") == filled_slots.get("to_currency"):
        continue

      args = {}
      for param_name in schema["parameters"]["properties"]:
        if param_name in filled_slots:
          args[param_name] = filled_slots[param_name]
        elif param_name in defaults:
          args[param_name] = defaults[param_name]

      args_key = json.dumps(args, sort_keys=True)
      if args_key in seen_args:
        continue
      seen_args.add(args_key)

      instruction = tmpl_text.format(**filled_slots)

      examples.append({
        "id": f"gen-{idx:04d}",
        "instruction": instruction,
        "tool_schema": schema,
        "target_call": {"name": tool_name, "arguments": args},
        "ambiguous": False,
      })
      idx += 1
      generated_for_tool += 1

  rng.shuffle(examples)
  return examples


def main():
  rng = random.Random(SEED)
  examples = generate_examples(rng)

  out_path = Path("tool_calling/data/raw/generated_tool_calls.jsonl")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as f:
    for ex in examples:
      f.write(json.dumps(ex, sort_keys=True) + "\n")

  print(f"Generated {len(examples)} examples to {out_path}")


if __name__ == "__main__":
  main()
