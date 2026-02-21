from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사<br>한 목적으로 이용함으로써 발생한 손해<br>7. '
 '수의사의 치료상의 과오로 생긴 손해, 수의사 자격이 없는 자의 치료행위로 인<br>한 손해(수의사의 소견 및 처방에 의한 경우도 동일) '
 '및 그로 인하여 가중된<br>손해<br>8. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태<br>9. '
 '원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관<br>리에 대한 태만<br>10'),
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
 'indexing': {'chunk_id': 'chunk_001115',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
