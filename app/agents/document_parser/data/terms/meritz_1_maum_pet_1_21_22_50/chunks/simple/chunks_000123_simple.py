from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금융감독 원장에게 조정을 신청할 수 있으며, 분쟁조정 '
 '과정에서 계약자는 관계 법령이 정하는 바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본의 제공 또는 청취를 포함한 다)을 요구할 '
 '수 있습니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000123',
              'chunk_char_len': 157,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
