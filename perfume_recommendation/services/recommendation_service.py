import os
import json
from typing import List, Dict
from models.llm_client import GPTClient  # GPTClient 추가

class RecommendationService:
    def __init__(self):
        self.perfumes = None
        self.gpt_client = GPTClient()  # GPTClient 인스턴스 생성

    def load_data(self) -> bool:
        """향수 데이터를 로드합니다."""
        try:
            if os.path.exists('data/perfumes.json'):
                with open('data/perfumes.json', 'r', encoding='utf-8') as f:
                    self.perfumes = json.load(f)
                return True
            else:
                print("perfumes.json 파일이 존재하지 않습니다.")
                return False
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {str(e)}")
            return False

    def get_all_recommendations(self) -> List[Dict]:
        """모든 향수를 반환합니다."""
        if not self.perfumes:
            raise ValueError("향수 데이터가 로드되지 않았습니다.")
        return self.perfumes

    def filter_recommendations(self, user_input: str) -> List[Dict]:
        """사용자 입력을 기반으로 향수를 필터링합니다."""
        if not self.perfumes:
            raise ValueError("향수 데이터가 로드되지 않았습니다.")
        
        filtered_perfumes = []
        for perfume in self.perfumes:
            if user_input.lower() in perfume.get('description', '').lower():
                filtered_perfumes.append(perfume)
        return filtered_perfumes

    def recommend_perfumes(self, user_input: str) -> str:
        """사용자 입력에 맞는 향수를 추천합니다."""
        if not self.perfumes:
            raise ValueError("향수 데이터가 로드되지 않았습니다.")
        
        # 향수 데이터 텍스트를 준비합니다.
        perfumes_text = "\n".join([f"{perfume['name']}: {perfume['description']}" for perfume in self.perfumes])
        
        # GPT를 이용해 향수를 추천합니다.
        prompt = self.gpt_client.create_prompt(user_input, perfumes_text)
        gpt_response = self.gpt_client.get_response(prompt)
        
        return gpt_response
