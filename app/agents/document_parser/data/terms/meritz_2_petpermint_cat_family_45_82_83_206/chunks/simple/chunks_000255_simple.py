from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제6항에 따라 보험계약이 연장된 경우 계약자는 그 최초 연장된 날로부터 90일 이내에 그 계약을 취소할 수 있으며, 계약자가 '
 '연장된 보험계약을 취소하는 경우 회사는 최초연 장된 날 이후 계약자가 납입한 보험료 전액을 환급합니다. \uf000 제6항에 따라 '
 '보험계약이 연장된 경우 보험계약의 연장 일은 회사가 계약자의 재가입의사를 확인한 날(계약자 등이 회사에 보험금을 청구함으로써 계약자에게 '
 '연락이 닿아 회 사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000255',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
