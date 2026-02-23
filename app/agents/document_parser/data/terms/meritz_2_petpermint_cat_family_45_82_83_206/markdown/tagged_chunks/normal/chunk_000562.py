from langchain_core.documents import Document

chunk = Document(
    page_content=('- 판탈출증으로 추간판을 2마디이상(또는 1마디 추간판\n'
 '- 에 대해 2회 이상) 수술하고도 마미신경증후군이 발\n'
 '- 생하여 하지의 현저한 마비 또는 대소변의 장해가\n'
 '- 있는 경우\n'
 '- 13) “추간판탈출증으로 인한 뚜렷한 신경 장해”란 추간판\n'
 '- 탈출증으로 추간판 1마디를 수술하고도 신경생리검사\n'
 '- 에서 명확한 신경근병증의 소견이 지속되고 척추신경\n'
 '- 근의 불완전 마비가 인정되는 경우\n'
 '- 14) “추간판탈출증으로 인한 약간의 신경 장해”란 추간판\n'
 '- 탈출증이 확인되고 신경생리검사에서 명확한 신경근병\n'
 '- 증의 소견이 지속되는 경우'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000562',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
