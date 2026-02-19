from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자는 ｢금융소비자보호에 관한 법률｣ 제47조 및 관련 규정이 정하는 바에 따라 계약체결에 대한 회사의 법위반사 항이 있는 '
 '경우 계약체결일부터 5년 이내의 범위에서 계약 자가 위반사항을 안 날로부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하여 위법계약의 '
 '해지를 요구할 수 있습니 다. \uf000 회사는 해지요구를 받은 날부터 10일 이내 수락여부를 계약자에 통지하여야 하며, 거절할 때에는 '
 '거절 사유를 함 께 통지하여야 합니다. \uf000 계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르 지 않는 경우 해당 계약을 '
 '해지할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 76},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000146',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
