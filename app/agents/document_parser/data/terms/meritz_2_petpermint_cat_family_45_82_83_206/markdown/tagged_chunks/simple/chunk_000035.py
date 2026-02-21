from langchain_core.documents import Document

chunk = Document(
    page_content=('립보험료를 감액하거나 중도인출을 하는 경우 제1항의 만기\n'
 '환급금은 가입시점의 예상금액보다 감소할 수 있습니다.# 제11조(보험금 받는 방법의 변경)\uf000 계약자(보험금 지급사유 발생 후에는 '
 '보험수익자)는 회\n'
 '사의 사업방법서에서 정한 바에 따라 보험금의 전부 또는55일부에 대하여 나누어 지급받거나 일시에 지급받는 방법으\n'
 '로 변경할 수 있습니다.\n'
 '\uf000 회사는 제1항에 따라 일시에 지급할 금액을 나누어 지급\n'
 '하는 경우에는 나중에 지급할 금액에 대하여 평균공시이율\n'
 '을 연단위 복리로 계산한 금액을 더하며, 나누어 지급할 금'),
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
 'indexing': {'chunk_id': 'chunk_000035',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
