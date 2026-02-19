from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조(손해의 통지 및 조사)\n'
 '① 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체없이 그 내용을 회사에 알려야 합니다.\n'
 '1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고상황 및 이들 사항의 증인이 있을 경우 그 주소와 성명 '
 '2. 피해자로부터 손해배상청구를 받았을 경우 3. 피해자로부터 손해배상책임에 관한 소송을 제기받았을 경우'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 24},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000147',
              'chunk_char_len': 209,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
