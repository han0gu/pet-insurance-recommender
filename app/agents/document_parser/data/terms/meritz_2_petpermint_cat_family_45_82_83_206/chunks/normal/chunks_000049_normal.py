from langchain_core.documents import Document

chunk = Document(
    page_content=('【계약자가 2명 이상인 경우 】\n'
 '계약자가 2명 이상인 경우, 계약 전 알릴 의무, 보험료 납입의무 등 보험계약에 따른 계약자의 의무를 연대로 합니다.\n'
 '【연대】\n'
 '2인 이상이 공동으로 책임지는 것을 뜻하며, 각자가 채 무의 전부를 이행할 책임을 지되(지분만큼 분할하여 책 임을 지는 것과는 다름), '
 '다만 어느 1인의 이행으로 나 머지 사람들도 책임을 면하게 되는 것을 말합니다.\n'
 '제3관 계약자의 계약 전 알릴 의무 등'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 57},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000049',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
