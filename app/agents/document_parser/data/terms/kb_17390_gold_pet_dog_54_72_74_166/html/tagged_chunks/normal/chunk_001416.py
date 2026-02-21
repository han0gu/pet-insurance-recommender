from langchain_core.documents import Document

chunk = Document(
    page_content=('. 모든 피보험자 또는 모든 보험수익자가 소득세법 시행령 제107조(장애인의 범<br>위) 제1항에서 규정한 장애인인 '
 "보험</h1><br><table id='85' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>관 련 법 "
 '규</td><td>소득세법</td></tr><tr><td colspan="2">∙ 소득세법 시행령 제107조(장애인의 범위)에서 규정한 '
 '장애인 1. "장애인복지법"에 따른 장애인 및 "장애아동 복지지원법"에 따른 장애 아동 중 기획재정부령으로 정하는 사람 2'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001416',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
