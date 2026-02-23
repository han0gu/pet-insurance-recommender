from langchain_core.documents import Document

chunk = Document(
    page_content=("② 피상속인의 직계존속</p><br><p id='100' data-category='list' "
 "style='font-size:14px'>③ 피상속인의 형제자매 ④ 피상속인의 4촌 이내의 방계혈족</p><br><p id='101' "
 "data-category='paragraph' style='font-size:14px'>제13조(대표자의 지정)</p><br><p "
 "id='102' data-category='paragraph' style='font-size:14px'>\uf000 계약자 또는 "
 "보험수익자가</p><br><p id='103'"),
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
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
