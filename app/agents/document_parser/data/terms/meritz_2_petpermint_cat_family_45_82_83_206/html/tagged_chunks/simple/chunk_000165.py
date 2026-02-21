from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금, 계약자적립액 등을 산정하는 기준이 되<br>는 가입금액을 계약시 선택한 금액보다 적은 금액으로<br>줄이는 것을 말합니다.(이에 '
 "따라 보험료, 보험금 및 해<br>약환급금도 줄어듭니다)</p><br><p id='22' data-category='paragraph' "
 "style='font-size:16px'>\uf000 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경<br>우 계약자와 피보험자가 "
 '동일하지 않을 때에는 보험금 지급<br>사유가 발생하기 전에 피보험자가 서면(「전자서명법」 제2<br>조 제2호에 따른 전자서명이 있는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
