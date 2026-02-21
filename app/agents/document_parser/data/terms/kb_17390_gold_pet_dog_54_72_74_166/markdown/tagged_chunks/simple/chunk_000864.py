from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이상에 해당되는 때를 말한다.\n'
 '- 가) 언어평가상 자음정확도가 75%미만인 경우 법\n'
 '- 나) 언어평가상 표현언어지수 65 미만인 경우 ㆍ\n'
 '- 10) 말하는 기능의 장해는 1년 이상 지속적인 언어치료를 시행한 후 증상이 규정\n'
 '- 고착되었을 때 평가하며, 객관적인 검사를 기초로 평가한다.\n'
 '- 11) 뇌‧중추신경계 손상(정신‧인지기능 저하, 편마비 등)으로 인한 말하는 기\n'
 '- 능의 장해(실어증, 구음장애) 또는 씹어먹는 기능의 장해는 신경계‧정신\n'
 '- 행동 장해 평가와 비교하여 그 중 높은 지급률 하나만 인정한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000864',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
