from langchain_core.documents import Document

chunk = Document(
    page_content=('12) "추간판탈출증으로 인한 심한 신경 장해" 란 추간판탈출증으로 추간판을 2마 디 이상(또는 1마디 추간판에 대해 2회 이상) '
 '수술하고도 마미신경증후군이 발생하여 하지의 현저한 마비 또는 대소변의 장해가 있는 경우 13) "추간판탈출증으로 인한 뚜렷한 신경 장해" '
 '란 추간판탈출증으로 추간판 1마 디를 수술하고도 신경생리검사에서 명확한 신경근병증의 소견이 지속되고 척 추신경근의 불완전 마비가 인정되는 '
 '경우 14) "추간판탈출증으로 인한 약간의 신경 장해" 란 추간판탈출증이 확인되고 신 경생리검사에서 명확한 신경근병증의 소견이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 141},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000921',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
