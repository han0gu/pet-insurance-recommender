from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조(보험의 목적의 명부)\n'
 '계약자는 항상 보험의 목적의 명부를 비치하여 회사가 열람을 요구할 경우에는 이에 따라 야 합니다.\n'
 '제3조(예치보험료)\n'
 '예치보험료는 계약체결일 이전 1개월 동안 1일 평균 보험의 목적의 수에 정해진 보험요율 을 적용하여 계산합니다.\n'
 '제4조(보험료의 정산방법)\n'
 '보험료는 보험의 목적의 정보의 변경을 기초로 하여 다음과 같이 정산합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 39},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
