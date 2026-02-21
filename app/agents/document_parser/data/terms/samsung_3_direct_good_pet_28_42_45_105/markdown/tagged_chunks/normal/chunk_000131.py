from langchain_core.documents import Document

chunk = Document(
    page_content=('2년차 이자 = (100원 +10원)(※원금+1년차 이자) ×10% = 11원\n'
 '→ 2년 시점의 총 이자금액 = 10원 + 11원 =21원2. 평균공시이율: 전체 보험회사 공시이율의 평균으로, 기본계약의 계약체결시점의 '
 '평\n'
 '균공시이율을 말합니다. 이 평균공시이율은 금융감독원 홈페이지(www.fss.or.kr)의\n'
 '「업무자료/보험업무」 내 「보험상품자료」 에서 확인할 수 있습니다.\n'
 '3. 보장부분 적용이율: 보장보험료를 산출할 때 적용하는 이율을 말합니다.\n'
 '4. 해약환급금: 계약이 해지되는 때에 회사가 계약자에게 돌려주는 금액을 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
