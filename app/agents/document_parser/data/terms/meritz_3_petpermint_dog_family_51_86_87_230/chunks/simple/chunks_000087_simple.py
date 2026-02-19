from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 청약을 철회할 때에 이미 보험금 지급사유가 발생하였으 나 계약자가 그 보험금 지급사유가 발생한 사실을 알지 못 한 경우에는 '
 '청약철회의 효력은 발생하지 않습니다. \uf000 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경 우 회사가 이를 증명하여야 '
 '합니다.\n'
 '제21조(약관교부 및 설명의무 등)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 69},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 164,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
