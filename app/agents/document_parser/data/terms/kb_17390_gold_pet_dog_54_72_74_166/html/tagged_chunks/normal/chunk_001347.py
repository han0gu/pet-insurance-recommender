from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>수단을 활용한 보험수익자 의사표시의 확인방법 포함)</p><br><h1 id='222' "
 "style='font-size:16px'>성이 확보된 전자적</h1><br><p id='223' "
 "data-category='paragraph' style='font-size:14px'>약</p><br><p id='224' "
 "data-category='paragraph' style='font-size:14px'>관<br>제6조(보험금의 청구)<br>\uf000 "
 '피보험자 또는 지정대리청구인은 제1조에 정한 특별약관의'),
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
 'indexing': {'chunk_id': 'chunk_001347',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
