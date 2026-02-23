from langchain_core.documents import Document

chunk = Document(
    page_content=('| <table><thead></thead><tbody><tr><td>계약일</td><td>보장개시일</td></tr><tr><td '
 'colspan="2">1년 2024년 4월 10일 2025년 4월 10일</td></tr><tr><td>\uf000 제1항에서 '
 '"연간"이란 계약일로부터 매1년 단위로 계약해당일 전일까지</td><td>도래하는</td></tr></tbody></table> 기간을 '
 '의미합니다. |  |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000554',
              'chunk_char_len': 227,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
