from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조 (손해의 통지 및 조사)\n'
 '① 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체없이 그 내용을 회사 에 알려야 합니다.\n'
 '1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고상황 및 이들 사항의 증인이 있을 경우 그 주소와 성명 '
 '2. 피해자로부터 손해배상청구를 받았을 경우 3. 피해자로부터 손해배상책임에 관한 소송을 제기받았을 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000570',
              'chunk_char_len': 211,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
