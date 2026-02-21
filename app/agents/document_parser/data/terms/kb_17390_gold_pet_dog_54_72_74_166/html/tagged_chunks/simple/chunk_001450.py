from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 해당 반려동물의 질병의 상태 등에 대한 수의사의 소견에 따라서 다르<br>게 적용할 수 있습니다.<br>\uf000 제2항에도 '
 '불구하고 보험업법 제97조 제1항 제5호 및 동법 시행령 제43조의2 제1<br>항에 따른 보장내용 등이 비슷한 보험계약(이하 '
 '"유사계약"이라 합니다)이 계약<br>청약일 현재 유지중이거나, 계약 청약일 전 6개월 이내에 계약자 및 피보험자의<br>요구 또는 '
 '보험료 납입 연체로 해지된 경우 유사계약에서 정한 부담보 기간 종료<br>일 이내에서 계약의 부담보 기간을 적용하고, 유사계약에서 정한 '
 '질병과'),
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
 'indexing': {'chunk_id': 'chunk_001450',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
