from langchain_core.documents import Document

chunk = Document(
    page_content=('에서 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금\n'
 '을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재된 1일당 보상한도액을 한도\n'
 '로 합니다.# <지급보험금의 계산>{(피보험자가 부담한 수술 당일 의료비 -4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별\n'
 '약관에서 지급한 보험금) × 보상비율}과 보험증권에서 정한 1일당 보상한도액 중적은 금액<예시안내>[반려묘 '
 '수술비(치과및구강질환포함)(재가입형) 확대보장 계산]· 보험가입금액 : 반려묘 의료비(치과및구강질환포함)(재가입형) 10만원'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000548',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
