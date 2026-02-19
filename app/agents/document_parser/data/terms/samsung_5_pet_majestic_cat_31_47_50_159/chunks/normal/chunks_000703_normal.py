from langchain_core.documents import Document

chunk = Document(
    page_content=('<보장한도>\n'
 '보험금의 종류 | 지급기준 | 보상한도\n'
 '이물제거(내시경) 보험금 | 이물제거를 목적으로 내시경을 받은 경우 | 보험가입금액의 100%\n'
 '이물제거(구토유도약물) 보험금 | 이물제거를 목적으로 구토유도약물을 받은 경우 | 보험가입금액의 10%\n'
 '<지급보험금의 계산>\n'
 '{ ( 피보험자가 부담한 이물제거 치료 당일 의료비\n'
 '- 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 지급한 보험금 - 4-2. 반려묘 수술비(치과및구강질환포함) '
 '확대보장(재가입형) 추가특별약관 지급한 보험금 - 자기부담금 ) × 보상비율 }'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 113},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000703',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
