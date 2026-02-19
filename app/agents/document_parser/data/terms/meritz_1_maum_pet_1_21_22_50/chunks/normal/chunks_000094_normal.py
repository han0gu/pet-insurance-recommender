from langchain_core.documents import Document

chunk = Document(
    page_content=('【보장개시일】\n'
 '회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보험료를 받은 날을 말하나, 회사가 승낙하기 전이라도 청약과 함께 제1회 보험료를 '
 '받은 경우에는 제1회 보험료 를 받은 날을 말합니다. 또한, 보장개시일을 계약일로 봅니다.\n'
 '③ 회사는 제2항에도 불구하고 다음 중 한 가지에 해당되는 경우에는 보장을 하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 185,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
