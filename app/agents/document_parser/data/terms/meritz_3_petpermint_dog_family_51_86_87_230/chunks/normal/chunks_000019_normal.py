from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조(보험금을 지급하지 않는 사유)\n'
 '\uf000 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생 한 때에는 보험금을 지급하지 않습니다.\n'
 '① 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자 가 심신상실 등으로 자유로운 의사결정을 할 수 없는 상태에서 자신을 해친 '
 '경우에는 보험금을 지급합니다.\n'
 '【심신상실】\n'
 '정신병, 정신박약, 심한 의식장애 등의 심신장애로 인하 여 사물 변별 능력 또는 의사 결정 능력이 없는 상태를 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 55},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
