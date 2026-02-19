from langchain_core.documents import Document

chunk = Document(
    page_content=('【타인을 위한 계약】\n'
 '계약자가 다른 사람의 이익을 위해 자기의 이름으로 체결하는 보험계약을 말합니다.\n'
 '제19조(특별약관의 소멸)\n'
 '이 특별약관의 반려동물의 사망으로 이 특별약관에서 규정하는 보험금 지급사유가 더 이상 발생할 수 없는 경우에는 이 특별약관은 그 때부터 '
 '효력이 없습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 29},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000184',
              'chunk_char_len': 157,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
