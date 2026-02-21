from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 이해에 반하는 자로 지정하는 경우에는 해당 내용이 규약에 반영되어야 하며, 반영되지 않은 경\n'
 '- 우에는 별도 피보험자의 동의를 받아야 합니다.\n'
 '- ③ 회사는 계약자를 통해 단체의 규약이 제2항을 충족하고 있는 지 확인을 해야 하며, 계약자는 이에\n'
 '- 협조하여야 합니다.\n'
 '# 제3조(단체요율의 적용)- ① 제1조의 단체는 단체요율을 적용할 수 있습니다. 다만, 제3종 단체는 구성원이 명확하고 위험의 동\n'
 '- 질성이 확보되어야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000109',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
