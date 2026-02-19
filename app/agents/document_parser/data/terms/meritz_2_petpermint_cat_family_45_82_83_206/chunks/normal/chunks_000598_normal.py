from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 특정질병 | 분류코드 | 항목명\n'
 '피부질환 | AGA004 | 기타 비뇨기계 양성 신생물\n'
 'AGB004 | 기타 비뇨기계 악성 신생물\n'
 'AGC004 | 기타 비뇨기계 신생물 (양성 또는 악성이 불 확실한)\n'
 'OAA001 | 급성 신부전\n'
 'OAA002 | 신우 신염\n'
 'OAA003 | 수신증\n'
 'OAA004 | 만성 신장 질환 (신부전 포함)\n'
 'OAA005 | 신장 결석\n'
 'OAA006 OAA007 | 방광염 방광 결석\n'
 'OAA008 | 요도 폐색\n'
 'OAA009 | 요로 결석증\n'
 'OAA010 | 신경성 배뇨 이상'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 171},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['urinary']},
 'indexing': {'chunk_id': 'chunk_000598',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
