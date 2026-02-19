from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금융감 독원장에게 조정을 신청할 수 있으며, 분쟁조정 '
 '과정에서 계약자는 관계 법령이 정하 는 바에 따라 회사가 기록 및 유지·관리하는 자료의 열람(사본의 제공 또는 청취를 포 함한다)을 '
 '요구할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000161',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
