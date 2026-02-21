from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 만기환급금의 지급시기 도래 7일 이전에<br>지급 사유와 금액을 알리지 않은 경우, 지급사<br>유가 발생한 날의 다음 날부터 '
 "청구일까지의 기<br>간은 [보장]공시이율을 적용하여 계산한 이자를<br>지급합니다.</p><br><p id='27' "
 "data-category='list' style='font-size:16px'>2. 지급이자의 계산은 연단위 복리로 계산하며, "
 '일<br>자 계산합니다.<br>3'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000900',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
