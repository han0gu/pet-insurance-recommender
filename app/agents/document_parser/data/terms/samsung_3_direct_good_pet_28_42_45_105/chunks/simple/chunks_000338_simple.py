from langchain_core.documents import Document

chunk = Document(
    page_content=('66 / 181\n'
 '→ 2년 시점의 총 이자금액 = 10원 + 11원 = 21원\n'
 '2. 평균공시이율: 전체 보험회사 공시이율의 평균으로, 이 계약 체결 시점의 이율을 말합니다. 이 평균공시이율은 금융감독원 '
 '홈페이지(www.fss.or.kr)의 「업무자료/ 보험업무」 내 「보험상품자료」에서 확인할 수 있습니다. 3. 해약환급금: 계약이 '
 '해지되는 때에 회사가 계약자에게 돌려주는 금액을 말합니다. 4. 이미 납입한 보험료 : 계약자가 실제로 납입한 보험료를 말합니다.\n'
 '④ 기간과 날짜 관련 용어'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 67},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000338',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
