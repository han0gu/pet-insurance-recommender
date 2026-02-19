from langchain_core.documents import Document

chunk = Document(
    page_content=('제8조(손해배상청구에 대한 회사의 해결)\n'
 '① 피보험자가 피해자에게 손해배상책임을 지는 사고가 생긴 때에는 피해자는 이 약관에 의하여 회사 가 피보험자에게 지급책임을 지는 금액한도 '
 '내에서 회사에 대하여 보험금의 지급을 직접 청구할 수 있습니다. 그러나 회사는 피보험자가 그 사고에 관하여 가지는 항변으로써 피해자에게 '
 '대항할 수 있습니다. ② 회사가 제1항의 청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며, 회사의 요구가 있 으면 계약자 및 '
 '피보험자는 필요한 서류증거의 제출, 증언 또는 증인출석에 협조하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
