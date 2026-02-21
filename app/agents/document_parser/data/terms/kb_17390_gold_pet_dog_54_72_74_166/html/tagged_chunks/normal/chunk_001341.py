from langchain_core.documents import Document

chunk = Document(
    page_content=('따라<br>변경지정한 다음의 자(이하"지정대리청구인"이라 합니다)가 제6조(보험금의 청<br>구)에 정한 구비서류 및 특별한 사정이 '
 '있음을 증명하는 서류를 제출하고 회사의<br>승낙을 얻어 이 특별약관의 보험금 수익자의 대리인으로서 이 특별약관의 보험금<br>을 청구할 '
 '수 있습니다.<br>1'),
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
 'indexing': {'chunk_id': 'chunk_001341',
              'chunk_char_len': 163,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
