from langchain_core.documents import Document

chunk = Document(
    page_content=("회사는 보험기간이</h1><br><p id='75' data-category='paragraph' "
 "style='font-size:16px'>끝난 때에 만기환급금을 보험수익자에게 지급합니다.</p><br><p id='76' "
 "data-category='list' style='font-size:16px'>\uf000 회사는 계약자 및 보험수익자의 청구에 의하여 "
 '제1항에 의한 만기환급금을 지급하<br>는 경우 청구일부터 3영업일 이내에 지급합니다.<br>공<br>\uf000 회사는 제1항에 의한 '
 '만기환급금의 지급시기가 되면 지급시기 7일 이전에 그'),
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
 'indexing': {'chunk_id': 'chunk_000059',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
