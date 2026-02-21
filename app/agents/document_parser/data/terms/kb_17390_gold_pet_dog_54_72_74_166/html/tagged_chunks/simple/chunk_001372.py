from langchain_core.documents import Document

chunk = Document(
    page_content=('. 갱신전 보장특약의 보험료가 정상적으로 납입완료 되었을 것<br>제1항에 따라 자동 갱신되는 경우 보험계약 청약서에 기재된 사항 및 '
 "보험증권에</p><br><h1 id='20' style='font-size:14px'>\uf000</h1><br><p id='21' "
 "data-category='paragraph' style='font-size:14px'>회사가 승인한 사항에 대하여 변경이 생긴 경우에는 "
 '계약자 또는 피보험자가 서면<br>으로 그 사실을 회사에 알리고 보험증권에 확인을 받아야 합니다.<br>\uf000 알릴의무에 대하여는 '
 '보통약관 제1절'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001372',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
