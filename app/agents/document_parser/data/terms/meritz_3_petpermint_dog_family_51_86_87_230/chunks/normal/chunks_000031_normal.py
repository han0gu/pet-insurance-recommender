from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 제1항의 규정에 정한 지급기일 내에 보험금을 지 급하지 않았을 때(제2항의 규정에서 정한 지급예정일을 통 지한 경우를 '
 '포함합니다)에는 그 다음날부터 지급일까지의 기간에 대하여 【별표1(보험금을 지급할 때의 적립이율 계 산)】에서 정한 이율로 계산한 금액을 '
 '보험금에 더하여 지 급합니다. 그러나 계약자, 피보험자 또는 보험수익자의 책'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 57},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
