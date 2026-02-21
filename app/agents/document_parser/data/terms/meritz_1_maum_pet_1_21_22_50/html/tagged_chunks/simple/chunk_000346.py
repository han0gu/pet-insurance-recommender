from langchain_core.documents import Document

chunk = Document(
    page_content=('체결할 때 또는 계약체결 이후 다음 각 호의 1에<br>해당하는 자 중에서 보험금의 대리청구인(이하「지정대리청구인」이라 합니다)을 '
 '2인<br>이내(2인 지정시 대표대리인을 지정)에서 지정(제4조에 따른 변경지정 포함)할 수 있습<br>니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000346',
              'chunk_char_len': 133,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
