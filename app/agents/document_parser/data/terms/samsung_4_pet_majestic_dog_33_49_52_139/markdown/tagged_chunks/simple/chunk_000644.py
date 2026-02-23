from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하여 정당한 권리를 가진 사람에게 부담하는 손해에 대한 배상책임.\n'
 '- 5. 피보험자의 심신상실로 인한 배상책임\n'
 '- 6. 피보험자의 지시에 따른 배상책임\n'
 '- 7. 피보험자의 불법행위 또는 폭력행위로 인한 배상책임\n'
 '- 8. 티끌, 먼지, 석면, 분진 또는 소음으로 생긴 손해에 대한 배상책임\n'
 '- 9. 전자파, 전자장(EMF)으로 생긴 손해에 대한 배상책임\n'
 '- 10. 벌과금 및 징벌적 손해에 대한 배상책임\n'
 '- 11. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적\n'
 '- 으로 이용함으로써 발생한 손해'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000644',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
