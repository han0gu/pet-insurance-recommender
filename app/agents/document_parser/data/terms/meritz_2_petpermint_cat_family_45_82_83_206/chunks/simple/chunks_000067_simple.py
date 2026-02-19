from langchain_core.documents import Document

chunk = Document(
    page_content=('명기되어 있는 사항으로 보험금 지급사유가 발생하였을 때(계약자 또 는 피보험자가 회사에 제출한 기초자료의 내용 중 중 요사항을 고의로 '
 '사실과 다르게 작성한 때에는 계약을 해지할 수 있습니다) ⑤ 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회 를 주지 않았거나 계약자 '
 '또는 피보험자가 사실대로 알리는 것을 방해한 경우, 계약자 또는 피보험자에게 사실대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 '
 '권유했을 때'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 61},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000067',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
