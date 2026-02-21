from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제1항 제1호의 경우에는 그 노력을 하였더라면 손해를 방지 또는 경감할 수 있었던<br>금액<br>2. 제1항 제2호의 경우에는 '
 '제3자로부터 손해의 배상을 받을 수 있었던 금액<br>3'),
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
 'indexing': {'chunk_id': 'chunk_000245',
              'chunk_char_len': 105,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
