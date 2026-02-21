from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>병</p><p id='200' data-category='paragraph' "
 "style='font-size:14px'>반<br>에는 이 특별약관의 보험가입금액을 최초 1회에 한하여 "
 '천식지속상태(급성중증천식)<br>려<br>진단비로 보험수익자에게 지급합니다'),
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
 'indexing': {'chunk_id': 'chunk_000654',
              'chunk_char_len': 165,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
