from langchain_core.documents import Document

chunk = Document(
    page_content=('발생한 의료비에서 4-1. 반려묘의료비(치과및구강질환포함)(재가입형) 특별약관 및\n'
 '4-2. 반려묘 수술비(치과및구강질환포함) 확대보장(재가입형) 추가특별약관 지급보험\n'
 '금 합계액과 「자기부담금」 을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재\n'
 '된 이 특별약관의 보상한도액을 한도로 합니다.<지급보험금의 계산># { ( 피보험자가 부담한 MRI또는 CT촬영 당일 의료비- - '
 '4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 지급한 보험금'),
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
 'indexing': {'chunk_id': 'chunk_000625',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
