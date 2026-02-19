from langchain_core.documents import Document

chunk = Document(
    page_content=('않았을 때(제2항에서 정한 지급예정일을 통지한 경우를 포 함합니다)에는 그 다음날부터 지급일까지의 기간에 대하여 【별표1(보험금을 지급할 '
 '때의 적립이율 계산)】에서 정한 이율로 계산한 금액을 보험금에 더하여 지급합니다. 그러나 계약자, 피보험자 또는 보험수익자의 책임있는 '
 '사유로 지급 이 지연된 때에는 그 해당기간에 대한 이자는 더하여 지급 하지 않습니다. 다만, 회사는 계약자 등이 분쟁조정을 신청 했다는 '
 '사유만으로 이자지급을 거절하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 90},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000206',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
