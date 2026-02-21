from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 상시고용된 수의사의 범위, 신고방법, 처방전 발급 및 보존 방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 '
 '사항은 농림축산식품부령으로 정한다. | ① 수의사는 자기가 직접 진료하거나 검안하지 아니하고는 진단서, 검안서, 증 명서 또는 처방전을 '
 '발급하지 못하며, 「약사법」 제85조제6항에 따른 동물 용 의약품(이하 "동물용 의약품"이라 한다)을 처방·투약하지 못한다'),
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
 'indexing': {'chunk_id': 'chunk_000466',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
