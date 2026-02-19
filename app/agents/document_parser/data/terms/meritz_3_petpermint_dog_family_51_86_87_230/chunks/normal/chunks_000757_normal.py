from langchain_core.documents import Document

chunk = Document(
    page_content=('12) “추간판탈출증으로 인한 심한 신경 장해”란 추간 판탈출증으로 추간판을 2마디이상(또는 1마디 추간판 에 대해 2회 이상) '
 '수술하고도 마미신경증후군이 발 생하여 하지의 현저한 마비 또는 대소변의 장해가 있는 경우 13) “추간판탈출증으로 인한 뚜렷한 신경 '
 '장해”란 추간판 탈출증으로 추간판 1마디를 수술하고도 신경생리검사 에서 명확한 신경근병증의 소견이 지속되고 척추신경 근의 불완전 마비가 '
 '인정되는 경우 14) “추간판탈출증으로 인한 약간의 신경 장해”란 추간판 탈출증이 확인되고 신경생리검사에서 명확한 신경근병 증의 소견이'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 213},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000757',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
