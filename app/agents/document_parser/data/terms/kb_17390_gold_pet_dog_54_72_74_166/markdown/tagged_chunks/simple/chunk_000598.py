from langchain_core.documents import Document

chunk = Document(
    page_content=('- 치료를 직접적인 목적으로 시행한 저주파자극치료, 초음파 치료, 체외충격파 또\n'
 '- 는 플라즈마 물리치료를 말합니다.\n'
 '- \uf000 제1항 제7호에서 "항암약물치료"라 함은 수의사가 반려동물의 암의 치료를 직접\n'
 '- 적인 목적으로 화학요법 항암제 또는 Tyrosine kinase inhibitor(TKI) 표적항암\n'
 '- 제를 사용하여 시행한 치료(정맥, 피하 또는 경구 등을 통해서 투여하는 약제를\n'
 '- 사용한 치료를 포함합니다.)를 말합니다. 다만, 항암 치료가 아닌 다른 질환 치료'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000598',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
