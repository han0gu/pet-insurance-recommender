from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 목적으로 이용함으로써 발생한 손해\n'
 '- 7. 수의사의 치료상의 과오로 생긴 손해, 수의사 자격이 없는 자의 치료행위로 인\n'
 '- 한 손해(수의사의 소견 및 처방에 의한 경우도 동일) 및 그로 인하여 가중된\n'
 '- 손해\n'
 '- 상\n'
 '- 8. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태\n'
 '- 해\n'
 '- 9. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관\n'
 '- 리에 대한 태만\n'
 '- 10. 동물보호법 위반 등 동물학대에 기인하는 손해'),
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
 'indexing': {'chunk_id': 'chunk_000632',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
