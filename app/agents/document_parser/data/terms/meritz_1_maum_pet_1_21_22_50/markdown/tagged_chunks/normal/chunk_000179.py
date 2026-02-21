from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다.\n'
 '- ③ 보험회사는 계약자를 통해 단체의 규약이 제2항을 충족하고 있는 지 확인을 해야 하며,\n'
 '- 계약자는 이에 협조하여야 합니다.\n'
 '# 제3조(단체요율의 적용)① 제1조(계약의 적용 범위)에 해당하는 단체는 단체요율을 적용할 수 있습니다. 다만, 제3\n'
 '종 단체는 구성원이 명확하고 위험의 동질성이 확보되어야 합니다.- 37 -② 단체 구성원의 일부만을 대상으로 가입하는 경우에는 대상단체의 '
 '위험과 피보험단체의'),
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
 'indexing': {'chunk_id': 'chunk_000179',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
