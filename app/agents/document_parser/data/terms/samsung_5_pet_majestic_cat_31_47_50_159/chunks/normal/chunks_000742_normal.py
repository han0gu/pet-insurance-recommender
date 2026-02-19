from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 회사가 지급할 제1항에서 정한 보험금은 피보험자가 부담한 MRI 또는 CT 촬영 당일 발생한 의료비에서 4-1. '
 '반려묘의료비(치과및구강질환포함)(재가입형) 특별약관 및 4-2. 반려묘 수술비(치과및구강질환포함) 확대보장(재가입형) 추가특별약관 '
 '지급보험 금 합계액과 「자기부담금」 을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재 된 이 특별약관의 보상한도액을 한도로 '
 '합니다.\n'
 '<지급보험금의 계산>\n'
 '{ ( 피보험자가 부담한 MRI또는 CT촬영 당일 의료비'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 117},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000742',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
