from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 특정질병 | 분류코드 | 항목명\n'
 'AGB002 | 이행상피세포암종 (방광)\n'
 'AGA003 | 기타 방광의 양성 신생물\n'
 'AGB003 | 기타 방광의 악성 신생물\n'
 'AGC003 | 기타 방광의 신생물 (양성 또는 악성이 불확실한)\n'
 'AGA004 | 기타 비뇨기계 양성 신생물\n'
 'AGB004 | 기타 비뇨기계 악성 신생물\n'
 'AGC004 | 기타 비뇨기계 신생물 (양성 또는 악성이 불확실한)\n'
 'OAA002 | 신우 신염\n'
 'OAA003 | 수신증\n'
 'OAA005 | 신장 결석\n'
 'OAA006 | 방광염\n'
 'OAA007 | 방광 결석'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 197},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['urinary']},
 'indexing': {'chunk_id': 'chunk_000681',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
