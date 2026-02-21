from langchain_core.documents import Document

chunk = Document(
    page_content=('손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 가집\n'
 '니다.\n'
 '1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권\n'
 '2. 피보험자가 손해배상을 함으로써 대위 취득하는 것이 있을 경우에는 그 대위권\n'
 '② 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한 권리를 행사하거나 지키는 것\n'
 '에 관하여 조치를 하여야 하며, 또한 회사가 요구하는 증거 및 서류를 제출하여야 합니\n'
 '다. 이에 필요한 비용은 회사가 지급합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000145',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
