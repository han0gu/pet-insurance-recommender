from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기간을 의미합니다.\n'
 '- \uf000 반려동물(강아지) 일반조항 제22조(재가입) 제1항 및 제2항에 따라 재가입한 경\n'
 '- 우 또는 반려동물(강아지) 일반조항 제22조(재가입) 제5항에 따라 이 특별약관\n'
 '- 계약이 연장된 경우에는 종전 계약의 보험기간을 연장하는 것으로 보아 제5항을\n'
 '- 적용하지 않습니다.\n'
 '- \uf000 반려동물(강아지) 일반조항 제22조(재가입) 제1항 및 제2항에 따라 재가입한 경\n'
 '- \uf000\n'
 '- 우 또는 반려동물(강아지) 일반조항 제22조(재가입) 제5항에 따라 이 특별약관'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000583',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
