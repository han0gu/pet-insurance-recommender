from langchain_core.documents import Document

chunk = Document(
    page_content=('| 평균공시이율 | 전체 보험회사 공시이율의 평균으로, 이 계약 체결 시점의 이율을 말합니다. (금융감독원 '
 '홈페이지(www.fss.or.kr) 의 "업무자료-보험상품자료"에서 확인할 수 있습니다.) |\n'
 '| 해약환급금 | 계약이 해지되는 때에 회사가 계약자에게 돌려주는 금액을 말합니다. |\n'
 '| 이미 납입한 | 보험료 계약자가 실제 납입한 보험료를 말합니다. |\n'
 '4. 기간과 날짜 관련용어| 용 어 | 정 의 |\n'
 '| --- | --- |\n'
 '| 보험기간 | 계약에 따라 보장을 받는 기간을 말합니다. 정상적으로 영업하는 날을 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
