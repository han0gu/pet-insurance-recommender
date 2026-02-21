from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다.<br>\uf000 제1항 또는 제2항에서 지정한 전자적 주소를 사실과 다르게 알리거나 알리지 않는<br>경우에는 회사가 알고 '
 '있는 최근의 전자적 주소로 보험계약 안내자료를 교부함으<br>로써 회사의 보험계약 안내자료 제공의무를 다한 것으로 보며, 전자적 주소를 '
 "사</p><br><p id='59' data-category='list'></p><br><p id='60' "
 "data-category='paragraph' style='font-size:14px'>실과 다르게 알리거나 알리지 않아 발생하는 "
 '불이익은 계약자가'),
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
 'indexing': {'chunk_id': 'chunk_001402',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
