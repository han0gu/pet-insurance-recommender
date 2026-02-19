from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 이 약관에 따른 해약환급금은「보험료 및 해약환급금 산 출방법서」에 따라 계산합니다. \uf000 해약환급금의 지급사유가 '
 '발생한 경우 계약자는 회사에 해약환급금을 청구하여야 하며, 회사는 청구를 접수한 날부 터 3영업일 이내에 해약환급금을 지급합니다. '
 '해약환급금 지급일까지의 기간에 대한 이자의 계산은【별표1(보험금을 지급할 때의 적립이율 계산)】에 따릅니다. \uf000 회사는 '
 '경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 77},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000152',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
