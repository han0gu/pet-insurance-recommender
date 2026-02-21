from langchain_core.documents import Document

chunk = Document(
    page_content=('알리고 보험증권에 확인을 받아야 합니다.<br>\uf000 알릴의무에 대하여는 보통약관 제1절 일반조항 제15조(상해보험계약 후 알릴 '
 "의<br>무)를 적용합니다.</p><br><p id='22' data-category='list'></p><br><h1 id='23' "
 "style='font-size:14px'>제3조(자동갱신 적용)</h1><p id='24' data-category='paragraph' "
 "style='font-size:16px'>- 134 -</p><p id='25' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_001373',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
