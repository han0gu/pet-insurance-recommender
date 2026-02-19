from langchain_core.documents import Document

chunk = Document(
    page_content=('. 4. 가산이율 적용시 금융위원회 또는 금융감독원이 정당한 사유로 인정하는 경우에는 해당 기간에 대하여 가산이율을 적용하지 않습니다. '
 '5. 보험계약대출이율은 보험개발원이 공시하는 보험계약대출이율을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 48},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000258',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
