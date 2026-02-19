from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 평균공시이율: 전체 보험회사 공시이율의 평균으로, 이 계약 체결 시점의 이율을 말합니다. 이 평균공시이율은 금융감독원 '
 '홈페이지(www.fss.or.kr)의「업무자료/보 험업무」내「보험상품자료」에서 확인할 수 있습니다. 3. 보장부분 적용이율 : '
 '보장보험료를 산출할 때 적용하는 이율을 말합니다. 4. 해약환급금: 계약이 해지되는 때에 회사가 계약자에게 돌려주는 금액을 말합니다. '
 '5. 이미 납입한 보험료 : 계약자가 실제로 납입한 보험료를 말합니다.\n'
 '④ 기간과 날짜 관련 용어'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 31},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000006',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
