from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 제1항 제1호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에 회사\n'
 '는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반사실(계약해지 등의 원인이 되는\n'
 '위반사실을 구체적으로 명시)뿐만 아니라 계약 전 알릴 의무사항이 중요한 사항에 해당\n'
 '되는 사유를 “반대증거가 있는 경우 이의를 제기할 수 있습니다”라는 문구와 함께 계\n'
 '약자에게 서면 또는 전자문서 등으로 알려 드립니다. 또한 이 경우 계약 해지로 인하여\n'
 '회사가 환급하여야 할 보험료가 있을 때에는 제33조(보험료의 환급)에 따른 보험료를'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000058',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
