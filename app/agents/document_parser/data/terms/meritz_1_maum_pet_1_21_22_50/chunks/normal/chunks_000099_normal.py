from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자(보험수익자와 계약자가 다른 경우 보험수익자를 포함합니다)에게 납입최고 (독촉)기간 내에 연체보험료를 납입하여야 한다는 내용 '
 '2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 납입최고(독촉)기 간이 끝나는 날의 다음날에 계약이 해지된다는 내용'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
