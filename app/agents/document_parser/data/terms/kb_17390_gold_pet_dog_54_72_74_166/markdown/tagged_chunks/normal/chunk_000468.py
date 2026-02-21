from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서, 증명서 또는 처방전의 서식, 기재사항, 그 밖에 필요한 사항은 '
 '농림축산식품부령으로 정한다. ⑤ 제1항에도 불구하고 농림축산식품부장관에게 신고한 축산농장에 상시고용된 수의사는 해당 농장의 가축에게 '
 '투여할 목적으로 동물용 의약품에 대한 처방 전을 발급할 수 있다. 이 경우 상시고용된 수의사의 범위, 신고방법, 처방전 발급 및 보존 '
 '방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 사항은 농림축산식품부령으로 정한다. |'),
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
 'indexing': {'chunk_id': 'chunk_000468',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
