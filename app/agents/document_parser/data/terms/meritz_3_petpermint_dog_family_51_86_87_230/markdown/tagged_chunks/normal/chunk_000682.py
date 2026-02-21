from langchain_core.documents import Document

chunk = Document(
    page_content=('| 9) 뚜렷한 치매 : CDR 척도 3점 | 60 |\n'
 '| 10) 약간의 치매 : CDR 척도 2점 | 40 |\n'
 '| 11) 심한 뇌전증 발작이 남았을 때 | 70 |\n'
 '| 12) 뚜렷한 뇌전증 발작이 남았을 때 | 40 |\n'
 '| 13) 약간의 뇌전증 발작이 남았을 때 | 10 |\n'
 '# 나. 장해판정기준# 1) 신경계- 가) “신경계에 장해를 남긴 때”라 함은 뇌, 척수\n'
 '- 및 말초신경계 손상으로 “<붙임>일상생활 기본\n'
 '- 동작(ADLs) 제한 장해평가표”의 5가지 기본동작\n'
 '- 중 하나 이상의 동작이 제한되었을 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000682',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
