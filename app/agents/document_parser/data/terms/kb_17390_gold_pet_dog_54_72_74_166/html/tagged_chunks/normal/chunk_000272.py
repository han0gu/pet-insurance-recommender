from langchain_core.documents import Document

chunk = Document(
    page_content=("권리가 소멸됩니다.</h1><br><p id='89' data-category='paragraph' "
 "style='font-size:14px'>제32조(중대사유로 인한 해지)</p><br><p id='90' "
 "data-category='paragraph' style='font-size:14px'>\uf000 회사는 아래와 같은 사실이 있을 "
 "경우에는 안 날부터 1개월 이내에 계약을 해지할<br>수 있습니다.</p><br><p id='91' "
 "data-category='paragraph' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000272',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
