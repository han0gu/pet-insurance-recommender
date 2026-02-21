from langchain_core.documents import Document

chunk = Document(
    page_content=('전념하는 것을 말합니다.<br>\uf000 제1항의 "의료기관"이라 함은 의료법 제3조(의료기관) 제2항에서 정한 국내의 병<br>원이나 '
 "의원 또는 국외의 의료관련법에서 정한 의료기관을 말합니다.</p><br><p id='202' "
 "data-category='paragraph' style='font-size:16px'>제4조(특별약관의 소멸)</p><br><p "
 "id='203' data-category='paragraph' style='font-size:16px'>피보험자가</p><br><p "
 "id='204'"),
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
 'indexing': {'chunk_id': 'chunk_000486',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
