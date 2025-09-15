import requests

body = {
    "Per1": 1.07,
    "Per2": 0.58,
    "Per3": 0.48,
    "Per4": 0.7667,
    "Per5": 1.2333,
    "Per6": 1.9933,
    "Per7": 0.34,
    "Per8": 1.01,
    "Per9": 0.8633,
    "Dem1": 0.46,
    "Dem2": 0.6433,
    "Dem3": 0.7366,
    "Dem4": 0.7566,
    "Dem5": 0.8133,
    "Dem6": 0.6933,
    "Dem7": 0.6666,
    "Dem8": 0.68,
    "Dem9": 0.7266,
    "Cred1": 0.6066,
    "Cred2": 1.01,
    "Cred3": 0.9333,
    "Cred4": 0.6033,
    "Cred5": 0.6866,
    "Cred6": 0.6733,
    "Normalised_FNT": -245.75
}

response = requests.post(
    url="http://127.0.0.1:8000/score",
    json=body
)

print(response.json())
 