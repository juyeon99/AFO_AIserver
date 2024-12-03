import json , logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class LineMapping:
    def __init__(self, line_file_path: str):
        self.line_mapping = self._load_line_mapping(line_file_path)

    def _load_line_mapping(self, line_file_path: str) -> Dict[str, Dict[str, str]]:
        """
        Load the line mapping from a JSON file.
        """
        try:
            with open(line_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError("Loaded line mapping data is not a list.")
                line_mapping = {item['name']: item for item in data}
                logger.info("Line mapping loaded successfully.")
                return line_mapping
        except Exception as e:
            logger.error(f"Error loading line mapping from file: {e}")
            raise RuntimeError(f"Error loading line mapping from file: {e}")

    def get_line_id(self, line_name: str) -> Optional[int]:
        """
        Get the line ID for a given line name.
        """
        line_data = self.line_mapping.get(line_name)
        if line_data:
            return line_data.get("id")
        else:
            logger.warning(f"Invalid line name requested: {line_name}")
            return None

    def is_valid_line(self, line_name: str) -> bool:
        """
        Check if the given line name is valid.
        """
        return line_name in self.line_mapping

    def get_all_lines(self) -> Dict[str, Dict[str, str]]:
        """
        Get all lines from the line mapping.
        """
        return self.line_mapping
