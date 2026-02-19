from langchain_core.documents import Document

chunk = Document(
    page_content=('【핵연료물질】 사용된 연료를 포함합니다. 【핵연료물질에 의하여 오염된 물질】 원자핵 분열 생성물을 포함합니다.\n'
 '제3조(손해의 발생과 통지)\n'
 '① 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체 없이 그 내용을 회사에 알려야 합 니다.\n'
 '1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고 상황 및 이들 사항 의 증인이 있을 경우 그 주소와 '
 '성명 2. 피해자로부터 손해배상청구를 받았을 경우 3. 피해자로부터 손해배상책임에 관한 소송을 제기 받았을 경우'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000140',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
