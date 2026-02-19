from langchain_core.documents import Document

chunk = Document(
    page_content=('제7관 분쟁의 조정 등\n'
 '제39조(분쟁의 조정)\n'
 '\uf000 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금융감독원장에게 조정을 신청할 수 있으며, '
 '분쟁조정 과정에서 계약자는 관계 법령이 정하는 바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본 의 제공 또는 청취를 '
 '포함한다)을 요구할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 83},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000160',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
