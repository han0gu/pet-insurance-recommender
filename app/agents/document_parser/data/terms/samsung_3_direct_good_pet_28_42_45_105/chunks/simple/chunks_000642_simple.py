from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험계약을 체결할 때 피보험자의 건강상태가 회사가 정한 기준에 적합하지 않은 경우 2. 보험계약을 체결한 후 계약 전 알릴 의무 '
 '위반의 효과 등으로 보장을 제한할 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000642',
              'chunk_char_len': 96,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
