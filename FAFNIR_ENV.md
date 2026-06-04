# HandyRL Fafnir Environment

## About Fafnir
[Farnir by Oink Games](https://oinkgames.com/games/analog/fafnir/)

## How to use

### Train
```
.venv\Scripts\python main.py --train 
```

### Use as AI Bot
```
.venv\Scripts\python fafnir_core/clients/ai_bot_handyrl.py --url http://127.0.0.1:8765/ --room room1 --name AI_HandyRL --model models/latest.pth
```

## Note
0~500 TD
500~1000
policy_target:UPGO
value_target:VTRACE
1000~ VTRACE