from langchain_core.documents import Document

chunk = Document(
    page_content=('- 7. 피보험자와 타인간에 손해배상에 관한 약정이 있는 경우 그 약정에 따라 가중된 배상책\n'
 '- 임\n'
 '- 8. 피보험자가 소유, 사용 또는 관리하는 재물이 손해를 입었을 경우에 그 재물에 대하여\n'
 '- 정당한 권리를 가진 사람에게 부담하는 배상책임\n'
 '- 9. 피보험자의 심신상실로 인한 배상책임\n'
 '- 10. 피보험자의 지시에 따른 배상책임\n'
 '- 11. 벌과금 및 징벌적 손해에 대한 배상책임\n'
 '- 12. 피보험자와 세대를 같이하는 친족에 대한 배상책임'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000128',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
