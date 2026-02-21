from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나) 언어평가상 표현언어지수 65 미만인 경우\n'
 '- 10) 말하는 기능의 장해는 1년 이상 지속적인 언어치료를\n'
 '- 시행한 후 증상이 고착되었을 때 평가하며, 객관적인\n'
 '- 검사를 기초로 평가한다.\n'
 '- 11) 뇌‧중추신경계 손상(정신‧인지기능 저하, 편마비 등)\n'
 '- 으로 인한 말하는 기능의 장해(실어증, 구음장애)\n'
 '- 또는 씹어먹는 기능의 장해는 신경계‧정신행동 장해\n'
 '- 평가와 비교하여 그 중 높은 지급률 하나만 인정한\n'
 '- 다.\n'
 '- 12) “치아의 결손”이란 치아의 상실 또는 발치된 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000616',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
