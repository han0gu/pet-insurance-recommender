from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 보험기간이 끝난 때에 만기환급금(중도인출이 있 는 경우에는 중도인출 원금과 이자를 차감하고 적립한 금액 을 '
 '말합니다)을 보험수익자에게 지급합니다. \uf000 회사는 계약자 및 보험수익자의 청구에 따라 제1항에 따 른 만기환급금을 지급하는 경우 '
 '청구일부터 3영업일 이내에 지급합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 55},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
