import torch
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers import pipeline
from PIL import Image
import requests
import re
import random

# GPT-J 모델 및 토크나이저 로드
# GPT-J가 사용되지 않는 경우 GPT2로 우회해서 코드실행
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 감정 분석 모델 로드
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 사전 정의된 응답 및 메뉴 정보
predefined_responses = {
    "en": {
        "intro": "Hello! I am CKDbot. Our store is OOO branch and our signature menu is XXX.",
        "menu": "Let me introduce the menu.\n Americano: KRW 5,000 \n (Popular) Cappuccino: KRW 4,000 \n (Recommended) XXX: KRW 6,000",
        "help": "introduce / menu / help / order / cart / url to check our store!",
        "history": "Here's a summary of our conversation:\n",
        "language_changed": "Language changed to English.",
        "order_added": "Item added to cart.",
        "cart": "Your current cart:",
        "url": "Here's our store URL: https://www.example.com/ooo-branch",
    },
    "ko": {
        "소개": "안녕하세요! 저는 CKD봇입니다. 저희 매장은 OOO점으로 시그니처 메뉴는 XXX으로 6000원입니다.",
        "메뉴": "메뉴 소개를 해드리겠습니다.\n 아메리카노 : 5,000원 \n (인기) 카푸치노 : 4,000원 \n (추천) XXX : 6,000원",
        "도움말": "소개 / 메뉴 / 기록 / 주문 / 장바구니 / URL을 통해 저희 매장을 확인해보세요!",
        "기록": "지금까지의 대화 내용입니다:\n",
        "language_changed": "언어가 한국어로 변경되었습니다.",
        "order_added": "장바구니에 항목이 추가되었습니다.",
        "장바구니": "현재 장바구니 내역:",
        "url": "저희 매장 URL입니다: https://www.example.com/ooo-branch",
    }
}

menu = {
    "아메리카노": 5000,
    "카푸치노": 4000,
    "XXX": 6000
}

max_context_length = 1024
max_history_length = 10  # 최대 기억할 대화 턴 수

cart = {}
user_preferences = {}

@torch.no_grad()
def generate_response(prompt, history, language, user_id, max_new_tokens=100):
    # NLU: 의도 파악
    intent = detect_intent(prompt, language)
    
    # 사전 정의된 응답 처리
    if intent in predefined_responses[language]:
        if intent in ["history", "기록"]:
            return predefined_responses[language][intent] + "\n".join(history)
        elif intent in ["cart", "장바구니"]:
            return predefined_responses[language][intent] + "\n" + display_cart(language)
        elif intent in ["url", "URL"]:
            return predefined_responses[language][intent]
        return predefined_responses[language][intent]
    
    # 주문 처리
    order_match = re.search(r'주문\s+(\S+)\s+(\d+)개', prompt)
    if order_match:
        item, quantity = order_match.groups()
        return add_to_cart(item, int(quantity), language)

    # 맥락 유지를 위한 전체 대화 내용 구성
    full_prompt = "\n".join(history) + "\n" + prompt if history else prompt
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    
    if input_ids.shape[1] > max_context_length:
        input_ids = input_ids[:, -max_context_length:]
    
    try:
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        
        # 개인화: 사용자 선호도 반영
        personalized_response = personalize_response(generated_response, user_id)
        
        # 감정 분석 및 응답 조정
        sentiment = analyze_sentiment(prompt)
        adjusted_response = adjust_response_based_on_sentiment(personalized_response, sentiment)
        
        return adjusted_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, there was a problem generating a response." if language == "en" else "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."

def detect_intent(prompt, language):
    # 간단한 키워드 기반 의도 탐지
    keywords = {
        "en": {"menu": "menu", "order": "order", "help": "help", "history": "history", "cart": "cart", "url": "url"},
        "ko": {"메뉴": "메뉴", "주문": "주문", "도움말": "도움말", "기록": "기록", "장바구니": "장바구니", "url": "url"}
    }
    
    for intent, keyword in keywords[language].items():
        if keyword in prompt.lower():
            return intent
    return "general"

def add_to_cart(item, quantity, language):
    if item in menu:
        if item in cart:
            cart[item] += quantity
        else:
            cart[item] = quantity
        return predefined_responses[language]["order_added"]
    else:
        return "Item not found in menu." if language == "en" else "메뉴에 없는 항목입니다."

def display_cart(language):
    if not cart:
        return "Your cart is empty." if language == "en" else "장바구니가 비어있습니다."
    
    total = sum(menu[item] * quantity for item, quantity in cart.items())
    cart_display = "\n".join(f"{item}: {quantity}" for item, quantity in cart.items())
    return f"{cart_display}\nTotal: {total}원"

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label']

def adjust_response_based_on_sentiment(response, sentiment):
    if sentiment == 'POSITIVE':
        return response + " I'm glad you're having a good experience!"
    elif sentiment == 'NEGATIVE':
        return "I'm sorry if there's been any inconvenience. " + response
    else:
        return response

def personalize_response(response, user_id):
    if user_id in user_preferences:
        favorite_item = user_preferences[user_id].get('favorite_item')
        if favorite_item:
            return f"Based on your preference for {favorite_item}, " + response
    return response

def update_user_preferences(user_id, item):
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
    user_preferences[user_id]['favorite_item'] = item

# 대화 루프
history = []
language = "ko"  # 기본 언어를 한국어로 설정
user_id = "user123"  # 실제 구현에서는 사용자 인증 시스템과 연동

print("OOO점을 방문해 주신 여러분 반갑습니다.\n 여러분을 도와줄 CKD봇입니다. 무엇을 도와 드릴까요?\n(종료하려면 'quit' 입력, 도움말은 '도움말', change language 'en')")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "취소"]:
        break
    elif user_input.lower() in ["en", "영어"]:
        language = "en"
        print(predefined_responses[language]["language_changed"])
        continue
    elif user_input.lower() in ["ko", "한글"]:
        language = "ko"
        print(predefined_responses[language]["language_changed"])
        continue
    
    prompt = f"Human: {user_input}\nAI:"
    response = generate_response(prompt, history, language, user_id)
    print(f"AI: {response}")
    
    # 실시간 상호작용: 랜덤하게 사용자 선호도 업데이트
    if random.random() < 0.2:  # 20% 확률로 선호도 업데이트
        random_item = random.choice(list(menu.keys()))
        update_user_preferences(user_id, random_item)
    
    history.append(f"Human: {user_input}")
    history.append(f"AI: {response}")
    
    # 히스토리 길이 제한
    if len(history) > max_history_length * 2:
        history = history[-max_history_length * 2:]

print("대화를 종료합니다." if language == "ko" else "Chat ended.")
