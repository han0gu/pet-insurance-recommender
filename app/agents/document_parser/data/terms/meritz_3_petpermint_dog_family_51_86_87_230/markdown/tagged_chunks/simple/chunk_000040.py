from langchain_core.documents import Document

chunk = Document(
    page_content=('규정에 의한 상속순위에 따라 상속받는 자를 말합니다.# 제14조(대표자의 지정)\uf000 계약자 또는 보험수익자가 2명 이상인 경우에는 '
 '각 대표\n'
 '자를 1명 지정하여야 합니다. 이 경우 그 대표자는 각각 다\n'
 '른 계약자 또는 보험수익자를 대리하는 것으로 합니다.\n'
 '\uf000 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은\n'
 '경우에는 이 계약에 관하여 회사가 계약자 또는 보험수익자\n'
 '1명에 대하여 한 행위는 각각 다른 계약자 또는 보험수익자\n'
 '에게도 효력이 미칩니다.\n'
 '\uf000 계약자가 2명 이상인 경우에는 그 책임을 연대로 합니'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
