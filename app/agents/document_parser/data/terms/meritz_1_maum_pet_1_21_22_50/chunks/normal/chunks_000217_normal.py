from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자는 매월 10일까지 전월말까지의 보험의 목적의 정보의 변경에 관한 서류를 회사 에 제출하여야 합니다. 그러나 계약이 효력상실 '
 '또는 해지된 경우에는 효력상실 또는 해지일까지의 보험료를 확정하기 위하여 필요한 서류를 효력상실 또는 해지 즉시 회사 에 제출하여야 '
 '합니다. 2. 회사는 보험기간 중이나 보험기간 만료후 보험료를 산출하기 위하여 필요하다고 인정될 경우에는 계약자의 서류를 열람할 수 '
 '있습니다. 3'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 39},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000217',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
