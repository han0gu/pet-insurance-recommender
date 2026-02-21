from langchain_core.documents import Document

chunk = Document(
    page_content=('취 등을 통해 확인받아야 하며, 설명서를 제공하여야 합니다.\n'
 '② 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제공 사실에 관하여 계약자와 회사\n'
 '간에 다툼이 있는 경우에는 회사가 이를 증명하여야 합니다.\n'
 '③ 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료(계약의 청약을 권유하\n'
 '기 위해 만든 자료 등을 말합니다)의 내용이 약관의 내용과 다른 경우에는 계약자에게'),
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
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 211,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
