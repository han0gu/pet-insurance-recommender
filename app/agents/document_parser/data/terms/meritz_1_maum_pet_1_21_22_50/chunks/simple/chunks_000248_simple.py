from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회사에 알리고 변경된 장애기간이 기재된 장애인증명서를 '
 '제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 46},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000248',
              'chunk_char_len': 86,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
