from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염<br>6. 피보험자의 피고용인이 피보험자의 업무에 종사중에 입은 신체의 장해로 인한 '
 '배상책임<br>7. 피보험자와 타인간에 손해배상에 관한 약정이 있는 경우 그 약정에 따라 가중된 배상책<br>임<br>8. 피보험자가 '
 '소유, 사용 또는 관리하는 재물이 손해를 입었을 경우에 그 재물에 대하여<br>정당한 권리를 가진 사람에게 부담하는 배상책임<br>9. '
 '피보험자의 심신상실로 인한 배상책임<br>10. 피보험자의 지시에 따른 배상책임<br>11'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
