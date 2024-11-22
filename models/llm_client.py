import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class GPTClient:
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()

        # 환경 변수에서 OPENAI_API_KEY 가져오기
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        # ChatOpenAI 인스턴스를 API 키와 함께 초기화
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

    def create_prompt(self, user_input: str, perfumes_text: str) -> str:
        """프롬프트를 생성합니다."""
        template = """
        당신은 향수 전문가입니다. 
        다음 조건에 맞는 응답을 답변하도록하세요.
        추론되는 시간은 10초 이내로 하세요.
        이름에 오 드 뚜왈렛 , 오 드 퍼퓸 은 부향률 이라는 뜻 입니다. 이름에 표시하지 않고
        부향률에 표시해주세요.
        조건: {user_input}

        향수 목록:
        {perfumes_text}

        **응답 예시**
        [추천계열] 추론된 계열이 무슨 계열과 매칭이 됬는지 어떤것인지 명시 , 부가 설명은 하지않음
        [추천느낌] 추천된 향수들이 어떤느낌을 주는지 설명해주세요.
        [추천향수] 조건과 가장 잘 맞는 3개의 향수를 추천해주세요. 이름 : , 부향률 (예시)오 드 퍼퓸 , 퍼퓸 , 오 드 뚜왈렛 : , 브랜드 : , 설명 : , 추천이유 :
        """
        return PromptTemplate(
            input_variables=["user_input", "perfumes_text"],
            template=template
        ).format(user_input=user_input, perfumes_text=perfumes_text)

    def get_response(self, prompt: str) -> str:
        """GPT 응답을 가져옵니다."""
        try:
            return self.llm.invoke(prompt).content
        except Exception as e:
            print(f"GPT 호출 중 오류 발생: {str(e)}")
            return "죄송합니다. 요청 처리 중 문제가 발생했습니다."
