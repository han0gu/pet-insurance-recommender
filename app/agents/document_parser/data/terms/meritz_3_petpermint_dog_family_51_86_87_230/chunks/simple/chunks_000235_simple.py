from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않습니다. 다만, 변경된 보험수익 자가 회사에 '
 '권리를 대항하기 위해서는 계약자가 보험수익 자가 변경되었음을 회사에 통지하여야 합니다.\n'
 '【부가설명】\n'
 '계약자가 보험수익자가 변경되었음을 회사에 통지하기 전에 보험금 지급사유가 발생한 경우 회사는 변경 전 보 험수익자에게 보험금을 지급할 수 '
 '있습니다. 회사가 변 경 전 보험수익자에게 보험금을 지급한 경우 변경된 보 험수익자에게는 별도로 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000235',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
