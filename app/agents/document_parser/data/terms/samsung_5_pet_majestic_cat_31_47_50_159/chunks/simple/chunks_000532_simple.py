from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 보험료 관련 용어\n'
 '1. 보험료: 손해를 보장하는데 필요한 보험료를 말합니다.\n'
 '⑥ 재가입 관련 용어\n'
 '1. 최초계약 : 최초로 체결되는 계약을 말합니다. 2. 재가입계약 : 이 보험의 사업방법서에서 정한 재가입 절차에 따라 재가입된 계약을 '
 '말합니다.\n'
 '제3조 (보험금의 지급사유)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 97},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000532',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
