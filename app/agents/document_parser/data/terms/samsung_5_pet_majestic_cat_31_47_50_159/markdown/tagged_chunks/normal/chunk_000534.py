from langchain_core.documents import Document

chunk = Document(
    page_content=('체결될 수 있습니다.\n'
 '1. 재가입일에 있어서 반려동물의 나이가 회사가 최초가입 당시 정한 재가입 나이의- 105 -# 범위 내일 것# 2. 재가입 전 계약의 '
 '보험료가 정상적으로 납입완료 되었을 것- ② 이 계약의 보험기간 종료 후 계약자가 재가입을 원하는 경우 계약자는 재가입 시점에\n'
 '- 서 회사가 판매하는 동일하거나 객관적이고 합리적인 범위내에서 기존 계약내용에 상\n'
 '- 응한 반려동물보험 상품(보험업감독규정 제1-2조(정의)에서 정한 장기손해보험에 한'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000534',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
