from langchain_core.documents import Document

chunk = Document(
    page_content=('① 사고가 발생하였을 경우 사고발생의 때와 곳, 피해자 의 주소와 성명, 사고상황 및 이들 사항의 증인이 있 을 경우 그 주소와 성명 ② '
 '피해자로부터 손해배상청구를 받았을 경우 ③ 피해자로부터 손해배상책임에 관한 소송을 제기 받았 을 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 176},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000587',
              'chunk_char_len': 133,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
