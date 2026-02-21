from langchain_core.documents import Document

chunk = Document(
    page_content=('- 중 하나 이상의 동작이 제한되었을 때를 말한다.\n'
 '- 나) 위 가)의 경우 “<붙임>일상생활 기본동작(ADLs) 제\n'
 '- 한 장해평가표”상 지급률이 10% 미만인 경우에는\n'
 '- 보장대상이 되는 장해로 인정하지 않는다.\n'
 '- 다) 신경계의 장해로 발생하는 다른 신체부위의 장해\n'
 '- (눈, 귀, 코, 팔, 다리 등)는 해당 장해로도 평가\n'
 '- 하고 그 중 높은 지급률을 적용한다.\n'
 '- 라) 뇌졸중, 뇌손상, 척수 및 신경계의 질환 등은 발\n'
 '- 병 또는 외상 후 12개월 동안 지속적으로 치료한\n'
 '- 후에 장해를 평가한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000683',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
