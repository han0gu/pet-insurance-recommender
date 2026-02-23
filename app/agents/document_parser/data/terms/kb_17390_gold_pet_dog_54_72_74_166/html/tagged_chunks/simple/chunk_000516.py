from langchain_core.documents import Document

chunk = Document(
    page_content=("치과의사 면허를</p><br><p id='252' data-category='list' style='font-size:14px'>가진 "
 '자(이하 "의사"라 합니다)에 의하여 치료가 필요하다고 인정한 경우로서 자<br>택 등에서 치료가 곤란하여 의료기관에서 의사의 관리 하에 '
 '직접적인 치료를 목<br>적으로 의료기구를 사용하여 생체(生體)에 절단(切斷), 절제(切除) 등의 조작(操<br>作)을 가하는 것을 '
 '말합니다.<br>\uf000 제1항의 수술에서 보건복지부 산하 신의료기술평가위원회(향후 제도 변경 시에는<br>동 위원회와 동일한 기능을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000516',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
