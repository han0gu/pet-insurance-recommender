from langchain_core.documents import Document

chunk = Document(
    page_content=('수술하고도 마미신경증<br>후군이 발생하여 하지의 현저한 마비 또는 대소변의 장해가 있는 경우<br>13) ‘추간판탈출증으로 인한 뚜렷한 '
 '신경 장해’란 추간판탈출증으로 추간<br>판 1마디를 수술하고도 신경생리검사에서 명확한 신경근병증의 소견이<br>지속되고 척추신경근의 '
 '불완전 마비가 인정되는 경우<br>14) ‘추간판탈출증으로 인한 약간의 신경 장해’란 추간판탈출증이 확인되<br>고 신경생리검사에서 '
 "명확한 신경근병증의 소견이 지속되는 경우</p><p id='63' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001565',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
