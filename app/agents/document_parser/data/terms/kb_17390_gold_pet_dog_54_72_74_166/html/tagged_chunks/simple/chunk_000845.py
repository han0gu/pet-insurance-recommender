from langchain_core.documents import Document

chunk = Document(
    page_content=('"반대증거가 있는 경우 이의를 제기할 수 있습니<br>다"라는 문구와 함께 계약자에게 서면 또는 전자문서 등으로 알려 '
 '드립니다.<br>회사가 전자문서로 안내하고자 할 경우에는 계약자에게 서면 또는 전자서명법 제2<br>조 제2호에 따른 전자서명으로 동의를 '
 '얻어 수신확인을 조건으로 전자문서를 송신<br>하여야 합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000845',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
