from langchain_core.documents import Document

chunk = Document(
    page_content=('가 필요로 하는 비용을 보전해 주기 위해 회사가 먼저 지급하는\n'
 '임시 교부금을 말합니다.\uf000 회사는 제1항의 규정에 정한 지급기일 내에 보험금을 지\n'
 '급하지 않았을 때(제2항의 규정에서 정한 지급예정일을 통\n'
 '지한 경우를 포함합니다)에는 그 다음날부터 지급일까지의\n'
 '기간에 대하여 【별표1(보험금을 지급할 때의 적립이율 계\n'
 '산)】에서 정한 이율로 계산한 금액을 보험금에 더하여 지\n'
 '급합니다. 그러나 계약자, 피보험자 또는 보험수익자의 책53임있는 사유로 지급이 지연된 때에는 그 해당기간에 대한'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000027',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
