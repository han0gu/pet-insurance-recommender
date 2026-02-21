from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약일은 제1회 보험료를 받은 날로 합니다.<br>\uf000 제1항 내지 제6항에도 불구하고 이 특별약관의 보험계약일로부터 그날을 '
 '포함하<br>여 90일 이내에 발생한 "백내장/녹내장수술", "특정약물치료Ⅱ", "항암약물치료</p><br><p id=\'213\' '
 "data-category='paragraph' style='font-size:16px'>- 112 -</p><p id='214' "
 'data-category=\'paragraph\' style=\'font-size:16px\'>" 또는 기타 이들과 유사한 사고에 '
 '대해서는 보험금을 지급하지'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001006',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
