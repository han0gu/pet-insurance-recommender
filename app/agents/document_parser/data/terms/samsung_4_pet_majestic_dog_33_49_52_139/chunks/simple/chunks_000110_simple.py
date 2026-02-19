from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '계약자가 회사에 보험수익자가 변경되었음을 통지하기 전에 보험금 지급사유가 발생한 경우 회사 는 변경 전 보험수익자에게 보험금을 지급할 수 '
 '있습니다. 회사가 변경 전 보험수익자에게 보험금 을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 43},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
