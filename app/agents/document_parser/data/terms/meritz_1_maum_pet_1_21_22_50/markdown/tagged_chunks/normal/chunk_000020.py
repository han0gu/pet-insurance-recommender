from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이용함으로써 발생한 손해\n'
 '- 9. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 치료행위\n'
 '- 로 인한 비용 및 그로 인하여 가중된 비용\n'
 '- 10. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태\n'
 '- 11. 대한민국 이외의 지역에서 발생한 사고 및 손해\n'
 '② 회사는 다음에 정한 사유 중 하나에 의해 피보험자가 부담한 치료비, 비용 또는 손해에\n'
 '대해서는 보험금을 지급하지 않습니다.- 1. 반려동물의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
