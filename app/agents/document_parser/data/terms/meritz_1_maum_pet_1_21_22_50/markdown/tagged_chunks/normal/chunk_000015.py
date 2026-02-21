from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑤ 제1항의 「연간」이라 함은 계약일부터 매 1년 단위로 도래하는 계약해당일 전일까지\n'
 '- 의 기간을 말합니다.\n'
 '- ➅ 반려동물이 제1항의 질병 또는 상해로 치료를 받던 중에 보험기간이 만료된 경우에도\n'
 '- 만료일부터 180일 이내의 치료비는 제2항에 따라 보상하여 드립니다. 다만, 사고일 또\n'
 '- 는 발병일부터 365일 이내인 경우에 한합니다.\n'
 '- ‡ 제1항의 「수술」이라 함은 수의사가 치료가 필요하다고 인정한 경우로서 수의사의 관\n'
 '- 리하에 치료를 직접적인 목적으로 기구를 사용하여 생체(生體)에 절단, 절제 등의 조작을 가'),
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
 'indexing': {'chunk_id': 'chunk_000015',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
