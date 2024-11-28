import json

class PromptLoader:
    def __init__(self, template_path: str):
        self.template_path = template_path
        self.template = self._load_template()

    def _load_template(self):
        """
        JSON 템플릿 파일을 로드합니다.
        """
        try:
            with open(self.template_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            raise ValueError(f"템플릿 파일을 로드하는 데 실패했습니다: {e}")

    def get_prompt(self, mode: str) -> dict:
        """
        대화 또는 추천에 해당하는 템플릿 섹션을 반환합니다.
        """
        if mode not in self.template:
            raise ValueError(f"{mode}에 해당하는 템플릿 섹션이 없습니다.")
        return self.template[mode]
