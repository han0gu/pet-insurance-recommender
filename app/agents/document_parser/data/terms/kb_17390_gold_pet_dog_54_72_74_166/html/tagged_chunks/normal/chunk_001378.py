from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 자동갱신을 원하지 않는 경우에는 갱신일에 갱신 전 보장특약<br>은 만료됩니다.<br>\uf000 제5항 및 '
 '제6항에도 불구하고, 회사가 계약자의 자동갱신 의사를 확인하지 못한<br>경우(계약자와 연락두절 등으로 회사 안내가 계약자에게 도달하지 '
 '못한 경우 포<br>함)에는 갱신일 현재의 약관 등으로 갱신됩니다'),
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
 'indexing': {'chunk_id': 'chunk_001378',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
