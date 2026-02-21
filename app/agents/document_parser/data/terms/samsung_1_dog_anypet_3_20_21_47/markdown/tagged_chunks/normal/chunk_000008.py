from langchain_core.documents import Document

chunk = Document(
    page_content=('2년 후\n'
 '단리계산법 : 원금 + (원금×10%) + (원금×10%) = 120원\n'
 '복리계산법 : 원금 + (원금×10%) + [원금 + (원금×10%)] ×10% = 121원나. 보험개발원이 공시하는 보험계약대출이율: '
 '보험개발원이 정기적으로 산출하여 공시하는 이\n'
 '율로써 회사가 보험금의 지급 또는 보험료의 환급을 지연하는 경우 등에 적용합니다.# 5. 기간과 날짜 관련 용어- 가. 보험기간: 계약에 '
 '따라 보장을 받는 기간을 말합니다.\n'
 "- 나. 영업일: 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일, '관공서의 공휴일에"),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000008',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
