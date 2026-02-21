from langchain_core.documents import Document

chunk = Document(
    page_content=('이율로 하며, 피보험자는 공탁금(이자를 포함합니다)의 회수청구권을 회사에 양도하여\n'
 '야 합니다.# 제14조(대위권)- ① 회사가 보험금을 지급한 때(현물보상한 경우를 포함합니다)에는 회사는 지급한 보험금\n'
 '- 의 한도내에서 아래의 권리를 가집니다. 다만, 회사가 보상한 금액이 피보험자가 입은\n'
 '- 손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 가집\n'
 '- 니다.\n'
 '- 1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
