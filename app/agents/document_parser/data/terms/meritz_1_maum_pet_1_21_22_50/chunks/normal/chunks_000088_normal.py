from langchain_core.documents import Document

chunk = Document(
    page_content=('【설명】\n'
 '계약자가 보험수익자가 변경되었음을 회사에 통지하기 전에 보험금 지급사유가 발생 한 경우 회사는 변경 전 보험수익자에게 보험금을 지급할 수 '
 '있습니다. 회사가 변경 전 보험수익자에게 보험금을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000088',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
