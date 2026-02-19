from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 보험증권: 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 드리는 증서를 말 합니다. 라. 갱신: 동일 '
 "보험상품('반려견보험 애니펫'), 유사보험상품(파밀리아리스 애견의료보험2) 또는 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 "
 '계약 중 회사가 유사하다고 판단 한 보험계약의 보험기간이 만료되어 종료일부터 7일 이내에 재가입하는 것을 말합니다.\n'
 '2. 지급사유 관련 용어'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000003',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
