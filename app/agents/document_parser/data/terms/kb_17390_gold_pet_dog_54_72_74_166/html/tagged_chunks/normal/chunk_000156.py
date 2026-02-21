from langchain_core.documents import Document

chunk = Document(
    page_content=('. “전문금융소비자”란 금융상품에 관한 전문성 또는 소유자산규모 등에 비<br>추어 금융상품 계약에 따른 위험감수능력이 있는 '
 '금융소비자로서 다음 각<br>목의 어느 하나에 해당하는 자를 말한다'),
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
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 107,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
