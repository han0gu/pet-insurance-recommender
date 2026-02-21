from langchain_core.documents import Document

chunk = Document(
    page_content=("id='54' data-category='paragraph' style='font-size:16px'>급금) 제1항에 따른 해약환급금을 "
 '계약자에게 지급합니다.<br>\uf000 제1항 제1호에 따른 계약의 해지가 보험금 지급사유 발<br>생 후에 이루어진 경우에 회사는 '
 '보험금을 지급하지 않으<br>며, 계약 전 알릴 의무 위반사실(계약해지 등의 원인이 되<br>는 위반사실을 구체적으로 명시)뿐만 아니라 '
 '계약 전 알릴<br>의무사항이 중요한 사항에 해당되는 사유를「반대증거가 있<br>는 경우 이의를 제기할 수 있습니다」라는 문구와 함께'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
