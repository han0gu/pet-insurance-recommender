from langchain_core.documents import Document

chunk = Document(
    page_content=("KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><p id='56' data-category='list'></p><br><p "
 "id='57' data-category='list'></p><br><p id='58' data-category='paragraph' "
 "style='font-size:14px'>회사에 알려야 합니다.<br>\uf000 제1항에서 지정한 전자적 주소가 변경되거나 사용 정지된 "
 '경우에는 그 사실을 지<br>체없이 회사에 알려야 합니다.<br>\uf000 제1항 또는 제2항에서 지정한 전자적 주소를 사실과 다르게 '
 '알리거나'),
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
 'indexing': {'chunk_id': 'chunk_001401',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
