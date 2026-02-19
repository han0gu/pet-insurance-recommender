from langchain_core.documents import Document

chunk = Document(
    page_content=('기간이 제1항의 지급기일을 초과할 것이 명백히 예상되는 경우에는 그 구체적인 사유와 지급예정일 및 보험금 가지급 제도(회사가 추정하는 '
 '보험금의 50% 이내를 지급)에 대하여 피보험자 또는 보험수익자에게 즉시 통지합니다. 다만, 지 급예정일은 다음 각 호의 어느 하나에 '
 '해당하는 경우를 제 외하고는 제7조(보험금의 청구)에서 정한 서류를 접수한 날 부터 30영업일 이내에서 정합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 53},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000027',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
